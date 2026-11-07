"""도서관 사서 보조 에이전트.

강의의 05_full_agent.py 구현 방식(도구, Runtime Context, 미들웨어,
가드레일, 단기/장기 기억, 최종 답변 검수)을 따른다.
외부 도서관 API와 RAG는 사용하지 않는다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langchain.agents.middleware import (
    AgentState,
    after_agent,
    before_agent,
    dynamic_prompt,
)
from langchain.agents.middleware.types import ModelRequest
from langchain.messages import AIMessage, HumanMessage
from langchain.tools import ToolRuntime, tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.runtime import Runtime
from langgraph.store.memory import InMemoryStore

load_dotenv()

MODEL = "qwen3:4b-instruct-2507-q4_K_M"

# ---------------------------------------------------------------------------
# 1. 도서 목록 (과제에서 제공한 목록만 사용)
# ---------------------------------------------------------------------------
BOOKS: list[dict[str, str]] = [
    {"title": "달빛 도서관의 비밀", "author": "최현범"},
    {"title": "내일이 사라진 도시", "author": "오다영"},
    {"title": "고양이는 자정에 책을 읽는다", "author": "김하린"},
    {"title": "시간을 파는 서점", "author": "이도윤"},
    {"title": "여름 끝에서 너를 만났다", "author": "박서연"},
    {"title": "기억을 잃은 탐정의 편지", "author": "정민재"},
    {"title": "열세 번째 문이 열리는 밤", "author": "한유진"},
    {"title": "구름 위 우체국의 마지막 편지", "author": "윤가람"},
    {"title": "마법사가 사라진 왕국", "author": "오지호"},
    {"title": "우리의 비밀은 별빛보다 깊다", "author": "강다은"},
    {"title": "사과의 매력을 찾아서", "author": "서정욱"},
]

CATALOG_TITLES = {book["title"] for book in BOOKS}

# 로컬 대출 원장. 도구로 확인하지 않은 값은 안내하지 않는다.
LOANS: dict[str, list[dict[str, str]]] = {
    "김민수": [
        {
            "title": "시간을 파는 서점",
            "due_date": "2026-11-20",
            "status": "대출중",
        }
    ],
    "이서준": [
        {
            "title": "달빛 도서관의 비밀",
            "due_date": "2026-11-10",
            "status": "연체",
        }
    ],
}

OFF_TOPIC_KEYWORDS = [
    "해킹",
    "폭탄",
    "주식 추천",
    "코인 투자",
    "숙제 대신",
    "코드 짜",
    "레시피",
    "요리법",
    "의료 진단",
    "처방",
    "연애 상담",
    "운세",
    "로또",
]

LIBRARY_KEYWORDS = [
    "책",
    "도서",
    "대출",
    "반납",
    "검색",
    "저자",
    "사서",
    "도서관",
    "열람",
    "연체",
    "예약",
    "선호",
    "제목",
    "작가",
]

OTHER_PATRON_LOAN_PATTERNS = [
    r"다른\s*(사람|이용자|회원|patron)",
    r"(전체|모든|아무)\s*(이용자|회원|사람).*(대출|대여)",
    r"(대출|대여).*(명단|목록|기록).*(전부|모두|전체)",
    r"누가\s*(빌려|대출)",
]


# ---------------------------------------------------------------------------
# 3. Runtime Context: 사서 이름·역할을 저장
# ---------------------------------------------------------------------------
@dataclass
class LibrarianContext:
    librarian_name: str
    librarian_role: str
    user_name: str
    user_id: str


# ---------------------------------------------------------------------------
# 2. 도구: 도서 검색, 대출 정보 확인, 이용자 회신, 선호 기억
# ---------------------------------------------------------------------------
@tool
def search_books(query: str) -> str:
    """도서 제목이나 저자 이름으로 소장 도서를 검색한다.

    Args:
        query: 검색할 도서 제목 또는 저자 이름.
    """
    keyword = query.strip()
    if not keyword:
        return "검색어가 비어 있습니다. 도서 제목이나 저자 이름을 입력해 주세요."

    lowered = keyword.lower()
    matches = [
        book
        for book in BOOKS
        if lowered in book["title"].lower() or lowered in book["author"].lower()
    ]
    if not matches:
        return f"'{keyword}'에 해당하는 도서를 찾을 수 없습니다."

    lines = [f"- {book['title']} / {book['author']}" for book in matches]
    return "검색 결과:\n" + "\n".join(lines)


@tool
def check_loan_info(
    runtime: ToolRuntime[LibrarianContext],
    patron_name: str | None = None,
) -> str:
    """현재 이용자의 대출 정보를 확인한다. 다른 이용자의 대출 정보는 조회하지 않는다.

    Args:
        patron_name: 조회 대상 이름. 비우면 현재 로그인한 이용자로 조회한다.
    """
    current_user = runtime.context.user_name
    requested = (patron_name or current_user).strip()

    if requested != current_user:
        return (
            "다른 이용자의 대출 정보는 공개할 수 없습니다. "
            f"현재 이용자({current_user})의 대출 정보만 확인할 수 있습니다."
        )

    records = LOANS.get(current_user, [])
    if not records:
        return f"{current_user} 님의 대출 중인 자료는 없습니다."

    lines = [
        f"- {item['title']} / 반납예정일 {item['due_date']} / 상태 {item['status']}"
        for item in records
    ]
    return f"{current_user} 님의 대출 정보:\n" + "\n".join(lines)


@tool
def reply_to_user(message: str) -> str:
    """이용자에게 전달할 공식 회신 문구를 정리한다. 도구로 확인한 내용만 담는다.

    Args:
        message: 이용자에게 안내할 회신 본문.
    """
    text = message.strip()
    if not text:
        return "회신할 내용이 없습니다."
    return f"[이용자 회신]\n{text}"


@tool
def save_user_preference(
    runtime: ToolRuntime[LibrarianContext],
    preference: str,
) -> str:
    """이용자의 도서 선호 등 장기 기억 정보를 저장한다.

    Args:
        preference: 저장할 선호 정보(장르, 작가, 안내 방식 등).
    """
    store = runtime.store
    if store is None:
        return "장기 기억 저장소를 사용할 수 없습니다."

    user_id = runtime.context.user_id
    existing = store.get(("preferences",), user_id)
    notes: list[str] = []
    if existing and isinstance(existing.value, dict):
        notes = list(existing.value.get("notes", []))
    if preference.strip() and preference.strip() not in notes:
        notes.append(preference.strip())
    store.put(("preferences",), user_id, {"notes": notes})
    return f"선호 정보를 저장했습니다: {preference.strip()}"


@tool
def recall_user_preference(runtime: ToolRuntime[LibrarianContext]) -> str:
    """저장된 이용자 선호 정보를 장기 기억에서 불러온다."""
    store = runtime.store
    if store is None:
        return "장기 기억 저장소를 사용할 수 없습니다."

    user_id = runtime.context.user_id
    existing = store.get(("preferences",), user_id)
    if not existing or not existing.value.get("notes"):
        return "저장된 이용자 선호 정보가 없습니다."
    notes = existing.value["notes"]
    return "저장된 선호 정보:\n" + "\n".join(f"- {note}" for note in notes)


# ---------------------------------------------------------------------------
# 3. 미들웨어: Runtime Context를 모델 입력(시스템 프롬프트)에 반영
# ---------------------------------------------------------------------------
@dynamic_prompt
def librarian_prompt(request: ModelRequest) -> str:
    ctx: LibrarianContext = request.runtime.context
    preference_text = "아직 저장된 선호 정보가 없습니다."
    store = request.runtime.store
    if store is not None:
        saved = store.get(("preferences",), ctx.user_id)
        if saved and saved.value.get("notes"):
            preference_text = "; ".join(saved.value["notes"])

    return f"""당신은 도서관 사서 보조 에이전트입니다.
사서 이름: {ctx.librarian_name}
역할: {ctx.librarian_role}
현재 이용자: {ctx.user_name} (id: {ctx.user_id})
이용자 선호(장기 기억): {preference_text}

규칙:
- 도서 검색, 대출 정보, 선호 정보는 반드시 도구로 확인한 뒤에만 안내한다.
- 도구로 확인하지 않은 도서, 대출 상태, 반납일, 소장 여부를 임의로 만들지 않는다.
- 검색 결과가 없으면 해당 도서를 찾을 수 없다고 안내한다.
- 도서관 업무와 무관한 요청은 정중히 거절한다.
- 다른 이용자의 대출 정보는 절대 공개하지 않는다.
- 최종 안내는 reply_to_user 도구로 정리한 내용을 바탕으로 답한다.
- 한국어로 답한다.
"""


# ---------------------------------------------------------------------------
# 4. 가드레일: 무관 요청 거절, 타 이용자 대출 정보 차단
# ---------------------------------------------------------------------------
def _latest_user_text(state: AgentState) -> str:
    for message in reversed(state.get("messages", [])):
        if isinstance(message, HumanMessage) or getattr(message, "type", "") == "human":
            content = message.content
            return content if isinstance(content, str) else str(content)
    return ""


@before_agent(can_jump_to=["end"])
def input_guardrail(state: AgentState, runtime: Runtime[LibrarianContext]) -> dict[str, Any] | None:
    text = _latest_user_text(state)
    if not text:
        return None

    if any(re.search(pattern, text) for pattern in OTHER_PATRON_LOAN_PATTERNS):
        return {
            "messages": [
                AIMessage(
                    content=(
                        "다른 이용자의 대출 정보는 공개할 수 없습니다. "
                        "본인 대출 현황만 안내할 수 있습니다."
                    )
                )
            ],
            "jump_to": "end",
        }

    lowered = text.lower()
    has_off_topic = any(keyword in text or keyword in lowered for keyword in OFF_TOPIC_KEYWORDS)
    has_library = any(keyword in text for keyword in LIBRARY_KEYWORDS)
    if has_off_topic and not has_library:
        return {
            "messages": [
                AIMessage(
                    content=(
                        "도서관 사서 보조는 도서 검색, 대출 안내, 이용자 선호 관리만 도와드릴 수 있습니다. "
                        "도서관 업무와 무관한 요청은 처리할 수 없습니다."
                    )
                )
            ],
            "jump_to": "end",
        }
    return None


# ---------------------------------------------------------------------------
# 6. 최종 답변 검수: 확인되지 않은 정보·부적절한 안내를 수정/차단
# ---------------------------------------------------------------------------
def _message_text(message: Any) -> str:
    content = getattr(message, "content", "")
    return content if isinstance(content, str) else str(content)


def _invented_titles(answer: str) -> list[str]:
    invented: list[str] = []
    quoted = re.findall(r"[「『\"']([^「『\"']{2,40})[」』\"']", answer)
    for title in quoted:
        if title not in CATALOG_TITLES and "찾을 수 없" not in answer:
            if any(token in title for token in ("책", "소설", "도서관", "서점", "편지", "왕국")):
                invented.append(title)
    return invented


@after_agent(can_jump_to=["end"])
def output_review(state: AgentState, runtime: Runtime[LibrarianContext]) -> dict[str, Any] | None:
    messages = state.get("messages", [])
    if not messages:
        return None

    last = messages[-1]
    if not isinstance(last, AIMessage):
        return None
    if getattr(last, "tool_calls", None):
        return None

    answer = _message_text(last)
    current_user = runtime.context.user_name

    for patron in LOANS:
        if patron != current_user and patron in answer:
            return {
                "messages": [
                    AIMessage(
                        content=(
                            "최종 답변에 다른 이용자의 정보가 포함되어 차단했습니다. "
                            "본인 대출 정보만 안내할 수 있습니다."
                        )
                    )
                ],
                "jump_to": "end",
            }

    tool_texts = [
        _message_text(message)
        for message in messages
        if getattr(message, "type", "") == "tool"
    ]
    loan_requested = "대출" in answer or "반납예정" in answer or "연체" in answer
    loan_confirmed = any("대출 정보" in text or "대출 중인 자료는 없습니다" in text for text in tool_texts)
    if loan_requested and not loan_confirmed:
        return {
            "messages": [
                AIMessage(
                    content=(
                        "확인되지 않은 대출 정보는 안내할 수 없습니다. "
                        "check_loan_info 도구로 확인한 뒤에만 안내합니다."
                    )
                )
            ],
            "jump_to": "end",
        }

    invented = _invented_titles(answer)
    if invented:
        return {
            "messages": [
                AIMessage(
                    content=(
                        "확인되지 않은 도서 정보가 포함되어 답변을 수정했습니다. "
                        "소장 목록에 없는 도서는 안내하지 않습니다. "
                        "제목이나 저자 이름으로 다시 검색해 주세요."
                    )
                )
            ],
            "jump_to": "end",
        }
    return None


def build_agent():
    model = ChatOllama(model=MODEL)
    checkpointer = InMemorySaver()
    store = InMemoryStore()
    agent = create_agent(
        model=model,
        tools=[
            search_books,
            check_loan_info,
            reply_to_user,
            save_user_preference,
            recall_user_preference,
        ],
        middleware=[librarian_prompt, input_guardrail, output_review],
        context_schema=LibrarianContext,
        checkpointer=checkpointer,
        store=store,
        name="library-librarian-assistant",
    )
    return agent, store


def default_context() -> LibrarianContext:
    return LibrarianContext(
        librarian_name="한서윤",
        librarian_role="도서관 사서 보조",
        user_name="김민수",
        user_id="patron-kim-minsu",
    )


def main() -> None:
    agent, store = build_agent()
    context = default_context()
    # 5. 장기 기억 초기 선호 예시
    store.put(
        ("preferences",),
        context.user_id,
        {"notes": ["따뜻한 분위기의 소설을 좋아함"]},
    )

    # 5. 단기 기억: 같은 thread_id로 대화 내용을 유지
    config = {"configurable": {"thread_id": f"library-{context.user_id}"}}
    print(
        f"{context.librarian_name} ({context.librarian_role})입니다. "
        f"{context.user_name} 님, 도서 검색·대출 안내를 도와드릴게요. 종료는 quit."
    )

    while True:
        user_input = input("\n이용자> ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit", "종료"}:
            print("이용을 종료합니다.")
            break

        result = agent.invoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config=config,
            context=context,
        )
        print(f"\n사서 보조> {result['messages'][-1].content}")


if __name__ == "__main__":
    main()
