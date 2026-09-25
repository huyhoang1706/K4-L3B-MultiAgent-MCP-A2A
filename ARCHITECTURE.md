# L3B Architecture Record

Tài liệu mô tả các quyết định có thể kiểm chứng trong source (`src/student_agent/`). Không chứa
prompt bí mật hay chain-of-thought; trace chỉ ghi sự kiện quan sát được.

## 1. System overview

Luồng chạy theo thứ tự cố định do **code** quyết định (pattern *deterministic flow* của OpenAI
Agents SDK — `examples/agent_patterns/deterministic.py`), LLM không tự chọn bước tiếp theo:

```text
                 case_received
                      │
                ┌─────▼──────┐   task_assigned    ┌──────────────┐
                │ Coordinator │──────────────────►│ Entity agent │  get_customer_history
                │ (get_policy)│◄──────────────────│              │  (+ get_order fallback)
                └─────┬──────┘      handoff        └──────────────┘
          task_assigned│ (song song)
     ┌─────────────────┼──────────────────┐
┌────▼──────┐   ┌──────▼───────┐   ┌──────▼────────┐
│Order/item │   │Payment/refund│   │  Shipment     │   ◄─ gọi MCP lấy evidence (least privilege)
│  agent    │   │    agent     │   │   agent       │      + LLM analyst gpt-6-luna đọc độc lập
└────┬──────┘   └──────┬───────┘   └──────┬────────┘
     └──── handoff ────┼──── handoff ─────┘
                ┌──────▼───────┐
                │ Policy agent │   ◄─ logic/rule: lỗi, bên chịu, số tiền (không gọi MCP)
                └──────┬───────┘
                handoff│
                ┌──────▼───────┐
                │Verifier agent│   ◄─ invariants + so sánh LLM vs rule + chấm confidence
                └──────┬───────┘
                       ▼
          outputs/<case_id>.json + traces/trace.jsonl (case_finalized)
```

So với khung 5 tầng của đề, L3B bổ sung **Entity agent** (resolve order/candidate, customer
history) trước khi fan-out, và Coordinator nạp policy theo `policy_version` của case để giao cho
Policy agent (Policy agent vẫn không gọi MCP).

| Module | Vai trò |
| --- | --- |
| `workflow.py` | Coordinator + điều phối 7 actor, emit trace lifecycle, dựng output |
| `evidence.py` | `EvidenceLedger`: quyền theo actor, cache theo case, retry có giới hạn, `tool_result_consumed` |
| `analysis.py` | Logic thuần (không I/O): timeline/anchor, findings, policy, claim, conflict |
| `llm_agents.py` | 3 Agents SDK analyst (`output_type` Pydantic) dùng `gpt-6-luna` |
| `mcp_gateway.py` | Client MCP; validate envelope theo `mcp-evidence-response-v1` |

## 2. Agent ownership

| Actor | Input | Trách nhiệm | Tool permission | Output/handoff |
| --- | --- | --- | --- | --- |
| Entity/customer (`entity-agent`) | `candidate_order_ids`, `claimed_order_id`, `customer_unique_id_hint` | Xác thực customer, xếp hạng/loại candidate, lấy toàn bộ instance của order | `get_customer_history`; `get_order` (chỉ fallback, tối đa 2 candidate) | `EntityResolution` → handoff `entity_resolved/ambiguous/not_found` cho coordinator |
| Coordinator | Case input, kết quả entity | Nhận case, giao việc, nạp policy, gom kết quả, không suy luận nghiệp vụ | `get_policy` | `task_assigned` cho 3 specialist; policy data cho Policy agent |
| Order/product (`order-agent`) | order_id đã resolve, timeline | Xác định anchor instance, trạng thái, item/seller, tổng tiền item | `get_order`, `get_order_items` | `OrderFinding` → handoff `order_finding_ready` |
| Shipment (`shipment-agent`) | order_id, timeline, OrderFinding | Đúng hạn/trễ, lỗi seller (handoff carrier > shipping limit) hay logistics, đối chiếu event actor; chỉ gọi summary khi claim liên quan giao hàng | `get_shipment_summary` | `ShipmentFinding` → handoff `shipment_<verdict>` |
| Payment/refund (`payment-agent`) | order_id, timeline, claim topic | Capture/mismatch/refund của anchor; refund timeline chỉ khi claim là refund | `get_payment_timeline`, `get_refund_timeline`, (`get_order_payments` dự phòng) | `PaymentFinding` → handoff `payment_finding_ready` |
| Policy (`policy-agent`) | 3 findings + policy rules + claim | Chọn primary issue, case status, action, refund, bên chịu trách nhiệm | Không (rule) | `PolicyDecision` → `policy_decided`, handoff cho verifier |
| Conflict resolver (trong Policy + `analysis.data_conflicts`) | Findings | Chọn nguồn theo precedence, ghi `data_conflicts`, tie-break theo claim khi instance trùng nhau | Không | `data_conflicts[]` trong output |
| Verifier (`verifier-agent`) | Output nháp, ledger, LLM readings | Kiểm tra invariant, so sánh LLM vs rule, calibrate confidence | Không (rule) | `verification_completed` (`passed`/`passed_with_warnings`) |

Least privilege được enforce trong code (`TOOL_PERMISSIONS`); actor gọi tool ngoài danh sách
sẽ bị `PermissionError`. Tool discovery (`list_tools`, cache một lần/session) chỉ dùng để kiểm
tra tool có tồn tại — không mở rộng quyền. `get_sellers`, `get_product_context` không được cấp
vì không đóng góp vào kết luận của 10 loại issue (tránh gọi thừa và evidence ngoài miền).

## 3. Entity resolution và A2A protocol

**Entity resolution**

1. Gọi `get_customer_history(customer_unique_id_hint)`; response phải có `customer_unique_id`
   trùng hint (scope check), nếu không bị loại.
2. Candidate có trong history ⇒ confirmed. Nhiều candidate confirmed ⇒ ưu tiên
   `claimed_order_id`; nếu claim không nằm trong số đó ⇒ `ambiguous`.
3. Không candidate nào khớp ⇒ fallback `get_order` cho tối đa 2 ID dạng order hợp lệ (32 hex,
   claimed trước). Candidate dạng `candidate-NNN` không bao giờ được query (tiết kiệm call).
4. Candidate còn lại ⇒ `rejected_candidates`. Confidence: 0.96 (xác thực qua history),
   0.72 (chỉ qua `get_order`), 0.45 (ambiguous), 0.2 (not found).
5. `not_found`/`ambiguous` ⇒ không điều tra tiếp, output `insufficient_evidence` +
   `needs_investigation` (không đoán).

**Anchor instance** (điểm mấu chốt của bộ dữ liệu L3B): một `order_id` có nhiều instance
(dòng history với purchase timestamp khác nhau). Instance được khiếu nại là instance **mua
muộn nhất ≤ `opened_at` mà kết quả đã quan sát được tại `opened_at`** (đã giao, hoặc quá hạn
dự kiến, hoặc canceled/unavailable). Event được gán cho instance theo: capture/mismatch → instance
có purchase gần nhất trước đó; `delivered_late` → instance có delivered date trùng event;
refund → instance đã capture đúng số tiền đó. `get_order` chỉ trả một dòng và thường là
instance khác, nên history được ưu tiên (xem mục 4).

**A2A message envelope**: mỗi bước là một trace event có `case_id` (correlation id), `actor`,
`target`, `decision_code` và `evidence_refs`; payload giữa agent là dataclass bất biến
(`EntityResolution`, `OrderFinding`, `PaymentFinding`, `ShipmentFinding`, `PolicyDecision`).
Điều kiện handoff: entity `resolved` mới fan-out; policy chỉ chạy khi cả 3 specialist đã trả
finding (kể cả finding "thiếu evidence"). Luồng là DAG cố định ⇒ không có vòng lặp; mỗi MCP
call có timeout 60 s, mỗi LLM analyst timeout 90 s. Trace không chứa suy luận riêng, prompt
hay nội dung khiếu nại.

## 4. Evidence và conflict lifecycle

1. `mcp_gateway` validate envelope theo `mcp-evidence-response-v1.schema.json`.
2. `EvidenceLedger._accept` kiểm tra scope: `order_id`/`customer_unique_id` trong data phải
   trùng tham số request; sai ⇒ bỏ, ghi failure `scope_mismatch`.
3. `evidence_ref` được lưu nguyên văn, không sửa/tạo; ledger là nguồn duy nhất của ref trong
   output. Mỗi lần một actor dùng kết quả ⇒ emit `tool_result_consumed` (tool, ref, domain,
   `result_hash`). Ledger sống trong phạm vi một case ⇒ evidence không bao giờ tái sử dụng
   giữa các case.
4. Chọn evidence theo claim: output chỉ trích dẫn history + order + policy và các domain
   hỗ trợ trực tiếp primary issue (vd. late delivery → shipment + items; refund → refund +
   payment timeline). Claim `requested_full_refund` liên kết payment/refund + policy.
5. Source precedence: history (authoritative, đủ instance) > `get_order` (một dòng) cho trạng
   thái/timeline của anchor; payment/refund lifecycle events > payment rows. Khác biệt được
   ghi trong `data_conflicts` với `selected_source`; conflict chưa giải quyết (vd. hai instance
   giống hệt nhau ủng hộ hai issue khác nhau, actor `delivered_late` trái với ngày handoff) có
   `selected_source: null` và resolution code mô tả, đồng thời hạ confidence.
6. Policy agent áp rule MCP policy theo `primary_issue` (status, action, refund, party type);
   `party_id` của seller lấy từ item của anchor (policy chỉ chứa seller ví dụ). Refund bị cap
   ở `refundable_total_brl`.

## 5. Failure and efficiency policy

| Failure | Retry budget | Fallback | Trace event/code |
| --- | ---: | --- | --- |
| HTTP 429 rate limit | 3 ở tầng HTTP (theo `Retry-After`, mặc định 2/5/10 s) | Hết lượt ⇒ xử lý như MCPError tạm thời | `mcp_failures` nếu vẫn lỗi |
| Lỗi transport httpx (reset, connect) | 2 retry connect của httpx; lỗi còn lại ⇒ 503 giả lập | `_GuardedTransport` biến lỗi thành lỗi của riêng request (không làm sập session/run) | như dòng dưới |
| MCPError tạm thời (`CONNECTION_CLOSED`, `REQUEST_TIMEOUT`, `INTERNAL_ERROR` = HTTP ≥ 400, thông điệp rate limit) | 2 (backoff 2 s, 5 s) | Hết lượt ⇒ domain thiếu, giảm confidence, không đoán | `verification_completed.attributes.mcp_failures`, reason `transient:mcp:<code>` |
| Timeout một call (60 s, sau khi được cấp slot) | 1 (call có thể đã được audit) | Như trên | reason `timeout` |
| MCP tool error (`is_error`, vd. không có refund lifecycle) | 0 (deterministic) | Coi như không có dữ liệu domain đó; claim refund thiếu refund lifecycle ⇒ `insufficient_evidence` | reason `tool_error` |
| Contract violation / scope mismatch | 0 | Bỏ evidence, không trích dẫn | `contract_violation` / `scope_mismatch` |
| Entity not found/ambiguous | ≤ 2 `get_order` fallback | `insufficient_evidence`, `needs_investigation`; không history ⇒ confidence ≤ 0.4, không bao giờ neo vào instance mua sau `opened_at` | handoff `entity_not_found` / `entity_ambiguous` |
| Thiếu order items | 0 | Không phân loại split/duplicate (không có tổng đơn); claim thanh toán ⇒ `insufficient_evidence`; seller lấy từ `shipping_limits` | `payment_finding_ready` |
| Source conflict | 0 | Precedence mục 4; tie-break theo claim khi instance trùng | `data_conflicts`, `policy_decided.attributes.claim_tiebreak` |
| Invalid specialist result / LLM lỗi | 0 (SDK tự retry transport) | Dùng finding rule-based; LLM chỉ ảnh hưởng confidence | handoff `llm_cross_check = skipped/disagree` |
| Invariant vi phạm | 0 | Giữ output, hạ confidence 0.1/check | `verification_completed = passed_with_warnings` |
| Session MCP sập (ExceptionGroup) | 0 | CLI báo số case chưa có output; chạy lại toàn bộ `day09 run` (một session) | lỗi CLI rõ ràng |

**Rate limit**: gateway có rate limit, nên mọi `tools/call` của mọi case đi qua một bộ giới hạn
chung (`_Pacer`): tối đa `DAY09_MCP_MAX_INFLIGHT` (mặc định 1) call đồng thời, các lần bắt đầu
cách nhau ≥ `DAY09_MCP_MIN_INTERVAL` (mặc định 0.5 s). Timeout 60 s chỉ tính từ lúc được cấp slot.

**Query budget (call plan)** — mặc định `lean`, **≤ 2 call/case**:

| Claim topic | Call 1 (entity-agent) | Call 2 (specialist) |
| --- | --- | --- |
| `late_delivery_seller`, `late_delivery_logistics` | `get_customer_history` | `get_shipment_summary` (actor, shipping limit, item/seller id, dòng order để phát hiện conflict) |
| `refund_pending`, `refund_failed` | `get_customer_history` | `get_refund_timeline` |
| còn lại (canceled, unavailable, split, duplicate, mismatch, unsupported) | `get_customer_history` | `get_payment_timeline` |

History luôn là call đầu tiên: resolve entity, customer context và chọn anchor instance (thứ quyết
định đúng/sai). Policy công khai `EC_POLICY_V2` giống hệt nhau ở mọi case nên được áp dụng từ
`policy_table.KNOWN_POLICIES` thay vì gọi `get_policy` mỗi case (nếu có gọi thì bản fetch luôn được
ưu tiên). Field nào không có evidence trong plan thì để `null`/`insufficient_evidence`, không đoán
(vd. `payment_analysis` của case late; `item_ids`/`seller_ids` khi không có items/shipment).
Split vs duplicate khi không có tổng đơn hàng là mơ hồ (cùng dạng 2 capture bằng nhau) ⇒ claim
quyết định, confidence −0.08. Mô phỏng offline trên 100 case: 200 call, primary issue đúng 100/100.

`--call-plan full` (5–6 call/case: thêm `get_order`, `get_order_items`, `get_policy`, shipment cho
claim giao hàng, refund cho claim refund) giữ nguyên hành vi của bài nộp đầu tiên (550 call).
Cache theo `(tool, args)` trong case, không quét candidate giả, không gọi
`get_sellers`/`get_product_context`. Retry idempotent (tool read-only) và không bao giờ biến
missing evidence thành dữ liệu phỏng đoán.

## 6. Verification invariants

Verifier kiểm tra trước khi finalize (`workflow._verify`):

- JSON Schema `l3b-output-v2` (CLI validate lần nữa trước khi ghi file);
- mọi `evidence_refs` ⊆ ref đã consumed trong case (ownership/provenance), không rỗng;
- `claim_assessments[].evidence_refs` ⊆ `evidence_refs` (claim linkage);
- entity scope: `resolved_order_ids ∩ rejected_candidates = ∅`;
- tổng `refund_lines` = `recommended_refund_brl`; refund ≤ `refundable_total_brl`;
- `no_action` ⇒ refund 0; refund > 0 ⇒ `action_required`;
- seller responsible ⇒ `party_id` thuộc `affected_entities.seller_ids`;
- `late_delivery_seller` ⇔ shipment `seller_delay` + party seller; `late_delivery_logistics` ⇔
  `logistics_delay` và không có seller;
- không trùng action;
- confidence ∈ [0.05, 0.97]: base 0.93; −0.2 khi tie-break theo claim; −0.05 instance trùng;
  −0.1 evidence chỉ ra lỗi khác claim; −0.1 conflict actor; −0.12 mỗi domain thiếu; −0.1 entity
  không xác thực qua history; −0.07 mỗi LLM analyst bất đồng; −0.1 mỗi invariant vi phạm.

## 7. Reproducibility

- Python ≥ 3.11 (đã chạy 3.12), dependencies trong `pyproject.toml` (`openai-agents` 0.22.x,
  `mcp` 2.x). Cài: `python -m pip install -e ".[dev]"`.
- LLM: `DAY09_LLM_MODEL=gpt-6-luna`, `DAY09_LLM_EFFORT=low`, `DAY09_LLM_CONCURRENCY=8`,
  `DAY09_LLM_TIMEOUT=90`; `DAY09_LLM=off` ⇒ chế độ rules-only cho kết quả nghiệp vụ giống hệt
  (LLM chỉ điều chỉnh confidence). Agents SDK tracing bật theo mặc định; tắt bằng
  `OPENAI_AGENTS_DISABLE_TRACING=1`.
- Không có random seed: logic nghiệp vụ là hàm thuần, deterministic.
- MCP: `DAY09_MCP_MAX_INFLIGHT=1`, `DAY09_MCP_MIN_INTERVAL=0.5`, `DAY09_MCP_CALL_TIMEOUT=60`.
- Lệnh: `day09 run --concurrency 2 --call-plan lean` → `day09 validate` → `day09 package --output dist/submission.zip`.
  Toàn bộ output phải sinh trong **một** lần `day09 run` (một MCP session) để mọi evidence ref
  thuộc cùng run.
- Debug cục bộ: `DAY09_DEBUG_DIR=debug` ghi finding/LLM reading từng case (không vào ZIP).
- Không ghi API key vào code, output, trace hay tài liệu.
