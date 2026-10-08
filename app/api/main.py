"""HTTP 接口层：SSE 流式对话 + 人工审批 + Trace 查询。

为什么售后 Agent 必须走 SSE 而不是普通 REST？
  因为 Agent 的"思考过程"是有价值的**过程信息**：
  客户要看到"正在为你查询物流…"，客服要看到"这一步被护栏拦了"。
  如果等 8 秒才返回一个 JSON，体验和排障都会很糟。
  SSE（Server-Sent Events）比 WebSocket 更合适：单向推送、自动重连、实现简单。

WebSocket 也一并提供了（/ws/chat），因为审批场景下客户端要回传消息。
"""

from __future__ import annotations

import asyncio
import json
import threading
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from ..config import settings
from ..db import Repo
from ..hitl import ApprovalManager
from ..obs.badcase import bad_case_report, explain
from ..obs.tracer import load_trace, load_traces
from ..router import describe_routing
from ..runner import AftersaleAgent
from ..runtime import doctor
from ..seed import ensure_seeded
from ..tools import all_tools, load_all, registry_summary

app = FastAPI(title="电商售后业务 Agent", version="1.0.0")

# 全局共享一个业务库连接（Repo 内部有锁，可跨线程）；避免每个请求都重连 SQLite
_repo = Repo()


# ==================================================================== 模型


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000, description="客户消息 / 工单内容")
    session_id: str = Field("", max_length=80, description="会话 ID，为空则新建")
    slots: dict[str, Any] = Field(default_factory=dict, description="已知槽位，例如 order_id")


class DecideRequest(BaseModel):
    approved: bool
    by: str = Field("人工客服", max_length=40)
    note: str = Field("", max_length=300)


# ==================================================================== 基础


@app.on_event("startup")
def _startup() -> None:
    load_all()
    ensure_seeded()


@app.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "capabilities": [
            {"name": item.name, "status": item.status, "real": item.real} for item in doctor()
        ],
        "llm": "mock" if settings.use_mock else settings.llm_model,
        "tools": len(all_tools()),
    }


@app.get("/tools")
def tools() -> dict:
    load_all()
    return {"count": len(all_tools()), "tools": registry_summary()}


@app.get("/intents")
def intents() -> dict:
    """意图路由表（纯文本，方便直接贴进文档/PPT）。"""
    return {"table": describe_routing()}


@app.get("/metrics")
def metrics() -> dict:
    with Repo() as repo:
        stats = repo.ticket_stats()
    runs = load_traces(limit=1000)
    total = len(runs) or 1
    return {
        "business": stats,
        "agent": {
            "runs": len(runs),
            "auto_close_rate": round(sum(1 for r in runs if r.get("auto_closed")) / total, 4),
            "handover_rate": round(sum(1 for r in runs if r.get("handover")) / total, 4),
            "blocked_rate": round(sum(1 for r in runs if r.get("blocked_calls")) / total, 4),
            "total_tokens": sum(int(r.get("total_tokens") or 0) for r in runs),
            "total_cost_yuan": round(sum(float(r.get("cost_yuan") or 0) for r in runs), 4),
        },
    }


# ==================================================================== 对话


def _agent(on_event=None) -> AftersaleAgent:
    return AftersaleAgent(repo=_repo, on_event=on_event)


@app.post("/chat")
async def chat(request: ChatRequest) -> dict:
    """非流式版本：适合脚本化调用与自动化测试。"""
    agent = _agent()
    result = agent.run(request.message, session_id=request.session_id, slots=request.slots)
    return result.to_dict()


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    """SSE 流式版本：把路由决策、每个节点、每次工具调用实时推给前端。"""
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def push(item: Any) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, item)

    def on_event(kind: str, payload: dict) -> None:
        push({"kind": kind, "payload": payload})

    def work() -> None:
        try:
            agent = _agent(on_event)
            result = agent.run(request.message, session_id=request.session_id,
                               slots=request.slots)
            push({"kind": "done", "payload": result.to_dict()})
        except Exception as exc:  # noqa: BLE001
            push({"kind": "error", "payload": {"error": f"{type(exc).__name__}: {exc}"}})
        finally:
            push(None)

    threading.Thread(target=work, daemon=True).start()

    async def generate():
        while True:
            item = await queue.get()
            if item is None:
                break
            yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",     # 关掉 nginx 缓冲，否则流式会被攒成一坨
    })


@app.websocket("/ws/chat")
async def ws_chat(websocket: WebSocket) -> None:
    """WebSocket 版：客户端可以连续发消息（多轮会话）。"""
    await websocket.accept()
    agent = _agent()
    try:
        while True:
            payload = json.loads(await websocket.receive_text())
            await websocket.send_json({"kind": "start", "payload": payload})
            result = await asyncio.to_thread(
                agent.run, payload.get("message", ""), payload.get("session_id", ""),
                "", payload.get("slots") or {},
            )
            await websocket.send_json({"kind": "done", "payload": result.to_dict()})
    except WebSocketDisconnect:
        return
    except Exception as exc:  # noqa: BLE001
        try:
            await websocket.send_json({"kind": "error", "payload": {"error": str(exc)}})
        except Exception:  # noqa: BLE001
            pass


# ==================================================================== 人工审批


@app.get("/approvals")
def approvals(status: str = "pending") -> dict:
    manager = ApprovalManager(_repo)
    items = manager.pending() if status == "pending" else manager.history()
    return {"count": len(items), "approvals": items}


@app.post("/approvals/{approval_id}/decide")
async def decide(approval_id: str, request: DecideRequest) -> dict:
    """审批并**自动恢复**被中断的图 —— 这是 HITL 的闭环。"""
    agent = _agent()
    record = agent.hitl.get(approval_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"审批单 {approval_id} 不存在")
    if record["status"] != "pending":
        raise HTTPException(status_code=409, detail=f"审批单已是 {record['status']} 状态")

    outcome = await asyncio.to_thread(
        agent.decide_approval, approval_id, request.approved, request.by, request.note
    )
    if not outcome.get("ok"):
        raise HTTPException(status_code=400, detail=outcome.get("error", "审批失败"))
    return outcome


@app.post("/approvals/sweep")
def sweep() -> dict:
    """把超时未处理的审批单按策略收掉（生产里由定时任务调用）。"""
    settled = ApprovalManager(_repo).sweep_timeouts()
    return {"settled": settled}


# ==================================================================== Trace


@app.get("/traces")
def traces(limit: int = 20) -> dict:
    items = load_traces(limit=limit)
    return {"count": len(items), "traces": items}


@app.get("/traces/{run_id}")
def trace_detail(run_id: str) -> dict:
    data = load_trace(run_id)
    if not data:
        raise HTTPException(status_code=404, detail=f"Trace {run_id} 不存在")
    return data


@app.get("/bad-cases")
def bad_cases() -> dict:
    return {"report": bad_case_report()}


@app.get("/explain/{run_id}")
def explain_run(run_id: str) -> dict:
    return {"report": explain(run_id)}


# ==================================================================== 演示页


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return _INDEX_HTML


_INDEX_HTML = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>电商售后业务 Agent</title>
<style>
:root{--bg:#0f1420;--panel:#171e2e;--line:#26304a;--fg:#e6ecf7;--dim:#93a1bd;--acc:#4f8cff;--ok:#37c98b;--warn:#f0b429;--err:#f2545b}
*{box-sizing:border-box}
body{margin:0;font:14px/1.6 -apple-system,"Segoe UI","Microsoft YaHei",sans-serif;background:var(--bg);color:var(--fg)}
header{padding:16px 22px;border-bottom:1px solid var(--line);display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px}
h1{font-size:16px;margin:0;font-weight:600}
.caps{display:flex;gap:6px;flex-wrap:wrap}
.cap{font-size:11px;padding:2px 8px;border-radius:10px;border:1px solid var(--line);color:var(--dim)}
.cap.real{color:var(--ok);border-color:#1d4d3a}
main{display:grid;grid-template-columns:1.4fr 1fr;gap:14px;padding:14px 22px;align-items:start}
@media(max-width:900px){main{grid-template-columns:1fr}}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px}
.card h2{font-size:13px;margin:0 0 10px;color:var(--dim);font-weight:600;letter-spacing:.5px}
textarea,input{width:100%;background:#0d1220;border:1px solid var(--line);color:var(--fg);border-radius:8px;padding:9px 11px;font:inherit;resize:vertical}
button{background:var(--acc);border:0;color:#fff;padding:9px 16px;border-radius:8px;cursor:pointer;font:inherit;font-weight:600}
button.ghost{background:transparent;border:1px solid var(--line);color:var(--fg);font-weight:400}
button:disabled{opacity:.5;cursor:not-allowed}
.row{display:flex;gap:8px;margin-top:10px;flex-wrap:wrap}
.examples{display:flex;gap:6px;flex-wrap:wrap;margin-top:10px}
.examples span{font-size:12px;padding:4px 9px;border:1px dashed var(--line);border-radius:14px;cursor:pointer;color:var(--dim)}
.examples span:hover{border-color:var(--acc);color:var(--fg)}
#log{max-height:460px;overflow:auto;font:12px/1.55 ui-monospace,Consolas,monospace}
.ev{padding:5px 8px;border-left:2px solid var(--line);margin-bottom:5px;background:#131a29;border-radius:0 6px 6px 0}
.ev.route{border-color:var(--acc)}.ev.tool{border-color:var(--ok)}.ev.guardrail{border-color:var(--err)}
.ev.approval{border-color:var(--warn)}.ev.error{border-color:var(--err)}
.ev b{color:var(--acc);font-weight:600}
pre{white-space:pre-wrap;word-break:break-word;margin:4px 0 0;color:var(--dim)}
.ap{border:1px solid var(--line);border-radius:8px;padding:10px;margin-bottom:8px}
.ap .t{font-weight:600}.ap .m{color:var(--dim);font-size:12px}
.empty{color:var(--dim);font-size:13px}
</style></head><body>
<header><h1>电商售后业务 Agent · 工具调用 + 工作流 + 人机协同</h1><div class="caps" id="caps"></div></header>
<main>
  <div>
    <div class="card"><h2>工单输入</h2>
      <textarea id="msg" rows="3" placeholder="例如：我要退款，订单号 SO2026000003"></textarea>
      <div class="examples">
        <span>帮我查下订单 SO2026000001 的物流到哪了</span>
        <span>我要退款，订单号 SO2026000003</span>
        <span>你们太差了！我要投诉，转人工</span>
        <span>我要退货，订单号 SO2026000001</span>
        <span>帮我分析一下工单各意图的占比</span>
      </div>
      <div class="row"><button id="send">提交工单（SSE 流式）</button>
      <button class="ghost" id="clear">清空</button></div>
    </div>
    <div class="card" style="margin-top:14px"><h2>执行过程（实时）</h2><div id="log"><div class="empty">等待提交…</div></div></div>
  </div>
  <div>
    <div class="card"><h2>待人工审批（关键动作）</h2><div id="approvals"><div class="empty">加载中…</div></div>
      <div class="row"><button class="ghost" id="refresh">刷新</button></div></div>
    <div class="card" style="margin-top:14px"><h2>结论</h2><div id="answer" class="empty">—</div></div>
    <div class="card" style="margin-top:14px"><h2>运行指标</h2><div id="metrics" class="empty">加载中…</div></div>
  </div>
</main>
<script>
const $=s=>document.querySelector(s);
const log=$('#log'); let session='sess-'+Math.random().toString(36).slice(2,9);
function addEv(cls,html){const d=document.createElement('div');d.className='ev '+cls;d.innerHTML=html;log.appendChild(d);log.scrollTop=log.scrollHeight;}
function esc(s){return (s??'').toString().replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
async function loadCaps(){const r=await fetch('/health');const d=await r.json();
  $('#caps').innerHTML=d.capabilities.map(c=>`<span class="cap ${c.real?'real':''}">${esc(c.name)}: ${esc(c.status.split('(')[0])}</span>`).join('');
  $('#caps').innerHTML+=`<span class="cap">工具 ${d.tools}</span>`;}
async function loadApprovals(){const r=await fetch('/approvals');const d=await r.json();
  const box=$('#approvals');
  if(!d.count){box.innerHTML='<div class="empty">当前没有待审批的关键动作。</div>';return;}
  box.innerHTML=d.approvals.map(a=>`<div class="ap"><div class="t">${esc(a.action)} · ¥${a.amount} <span class="m">[${esc(a.risk_level)}]</span></div>
    <div class="m">${esc(a.reason)}</div><div class="m">审批单 ${esc(a.approval_id)}</div>
    <div class="row"><button data-ap="${a.approval_id}" data-ok="1">批准并继续</button>
    <button class="ghost" data-ap="${a.approval_id}" data-ok="0">驳回</button></div></div>`).join('');}
async function loadMetrics(){const r=await fetch('/metrics');const d=await r.json();
  $('#metrics').innerHTML=`<div>工单总数 ${d.business.tickets_total}｜历史自动闭环 ${(d.business.auto_rate*100).toFixed(1)}%</div>
   <div>Agent 处理 ${d.agent.runs} 单｜全自动闭环 ${(d.agent.auto_close_rate*100).toFixed(1)}%｜转人工 ${(d.agent.handover_rate*100).toFixed(1)}%</div>
   <div>累计 tokens ${d.agent.total_tokens}｜成本 ≈¥${d.agent.total_cost_yuan}</div>`;}
async function decide(id,ok){
  addEv('approval',`<b>[人工决策]</b> ${ok?'批准':'驳回'} ${id} —— 正在唤醒被中断的图…`);
  try{
    const resp=await fetch(`/approvals/${id}/decide`,{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({approved:!!ok,by:'演示客服',note:ok?'已核对条款，同意':'依据不足，驳回'})});
    const data=await resp.json();
    const res=data.result||{};
    if(res.run_id) await replayRun(res.run_id);
    if(res.stopped_reason){
      addEv('approval',`<b>[恢复执行]</b> 结束=${esc(res.stopped_reason)}｜工具 ${res.tool_calls||0} 次｜拦截 ${res.blocked_calls||0} 次｜tokens ${(res.prompt_tokens||0)+(res.completion_tokens||0)}｜闭环=${res.auto_closed}`);
      if(res.answer)$('#answer').innerHTML=`<div>${esc(res.answer).replace(/\\n/g,'<br>')}</div>`;
    }else if(data.error){addEv('error',`<b>[恢复失败]</b> ${esc(data.error)}`);}
  }catch(e){addEv('error',`<b>审批请求失败</b> ${esc(e)}`);}
  loadApprovals();loadMetrics();}
/* 恢复执行本身是一次普通 POST（不是 SSE），所以审批后要把这次 run 里
   "中断之后"的节点/工具从 Trace 里捞出来补进左侧面板 —— 否则点了批准
   只能看到一行「已批准」，看不到图真的接着跑完了。 */
async function replayRun(runId){
  try{
    const t=await (await fetch(`/traces/${runId}`)).json();
    const ev=(t.events||[]).filter(e=>e.kind==='node'||e.kind==='tool'||e.kind==='approval');
    let cut=0;ev.forEach((e,i)=>{if(e.kind==='approval')cut=i+1;});
    for(const e of ev.slice(cut)){
      if(e.kind==='tool')addEv(e.blocked?'guardrail':'tool',`<b>[工具]</b> ${esc(e.name)}(${esc(JSON.stringify(e.arguments||{}))}) ${e.blocked?'被拦':'成功'} ${e.duration_ms||0}ms<pre>${esc((e.output||e.block_reason||'').slice(0,200))}</pre>`);
      else if(e.kind==='approval')addEv('approval',`<b>[审批]</b> ${esc(e.name)} — ${esc(e.decision||'')}`);
      else addEv('',`<b>[节点]</b> ${esc(e.name)} （${e.duration_ms||0}ms）`);
    }
  }catch(e){}
}
$('#send').onclick=async()=>{
  const message=$('#msg').value.trim(); if(!message)return;
  log.innerHTML=''; $('#answer').textContent='—'; $('#send').disabled=true;
  try{
    const resp=await fetch('/chat/stream',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({message,session_id:session})});
    const reader=resp.body.getReader(); const dec=new TextDecoder(); let buf='';
    while(true){const {value,done}=await reader.read(); if(done)break;
      buf+=dec.decode(value,{stream:true});
      const parts=buf.split('\\n\\n'); buf=parts.pop();
      for(const p of parts){ if(!p.startsWith('data: '))continue;
        const item=JSON.parse(p.slice(6)); render(item); } }
  }catch(e){addEv('error',`<b>请求失败</b> ${esc(e)}`);}
  finally{$('#send').disabled=false;loadApprovals();loadMetrics();}};
function render(item){
  const k=item.kind,p=item.payload||{};
  if(k==='route'){addEv('route',`<b>[路由]</b> ${esc(p.label)}（置信度 ${p.confidence}，来源 ${p.source}）→ ${esc(p.route)}${p.workflow?'/'+esc(p.workflow):''}`);}
  else if(k==='node'){addEv('',`<b>[节点]</b> ${esc(p.node)} （${p.duration_ms}ms）`);}
  else if(k==='step'){
    const t=p.type,d=p;
    if(t==='tool_call'){addEv('tool',`<b>[工具]</b> ${esc(d.tool)}(${esc(JSON.stringify(d.arguments))}) ${d.ok?'成功':'失败'} ${d.duration_ms}ms<pre>${esc((d.output||d.error||'').slice(0,300))}</pre>`);}
    else if(t==='guardrail'){addEv('guardrail',`<b>[护栏拦截]</b> ${esc(d.tool)} — ${esc(d.reason)}`);}
    else if(t==='approval'){addEv('approval',`<b>[待审批]</b> ${esc(d.tool)} — ${esc(d.reason)}`);}
    else if(t==='llm_call'){addEv('',`<b>[模型]</b> ${esc(d.model||'')} tokens ${d.prompt_tokens}+${d.completion_tokens}${d.downgraded?' [已降级]':''}<pre>${esc((d.text||'').slice(0,200))}</pre>`);}
  }
  else if(k==='done'){
    addEv('',`<b>[完成]</b> 结束=${esc(p.stopped_reason)}｜工具 ${p.tool_calls} 次｜拦截 ${p.blocked_calls} 次｜tokens ${(p.prompt_tokens||0)+(p.completion_tokens||0)}｜闭环=${p.auto_closed}`);
    $('#answer').innerHTML=`<div>${esc(p.answer).replace(/\\n/g,'<br>')}</div>
      <div class="m" style="color:#93a1bd;margin-top:8px;font-size:12px">意图 ${esc(p.intent)}｜路由 ${esc(p.route)}｜引擎 ${esc(p.engine)}｜审批 ${p.approvals}</div>`;}
  else if(k==='error'){addEv('error',`<b>[错误]</b> ${esc(p.error)}`);}
}
document.addEventListener('click',e=>{
  if(e.target.dataset.ap) decide(e.target.dataset.ap, e.target.dataset.ok==='1');
  if(e.target.classList.contains('examples')||e.target.parentElement?.classList.contains('examples')) $('#msg').value=e.target.textContent;});
$('#clear').onclick=()=>{log.innerHTML='<div class="empty">等待提交…</div>';$('#answer').textContent='—';};
$('#refresh').onclick=loadApprovals;
loadCaps();loadApprovals();loadMetrics();
setInterval(loadApprovals,15000);
</script></body></html>"""
