"""Unit and regression tests for citation verification and supporting sources filtering.

Covers the critical cases specified in Section 8:
- Answer with correct text but incorrect citation (must be rejected / not verified)
- Answer paraphrased from source but semantically supported (must be verified)
- Refusal to fall back to unverified chunks when evidence is missing
- Number mismatch rejection (claim numbers must match chunk)
- Out-of-scope question citation abstention
"""

import pytest

from app.rag.from_scratch.pipeline import _filter_supporting_sources, classify_answer_state
from app.rag.from_scratch.quality import evaluate_answer_response


@pytest.fixture
def sample_chunks() -> list[dict]:
    return [
        {
            "part": "2.1",
            "part_title": "QUY CHẾ ĐÀO TẠO ĐẠI HỌC",
            "chapter": "I",
            "chapter_title": "NHỮNG QUY ĐỊNH CHUNG",
            "article": 11,
            "article_title": "Điều 11. Xử lý kết quả học tập",
            "clauses": [1, 2],
            "page_start": 23,
            "page_end": 23,
            "text": (
                "1. Sau mỗi học kỳ chính, sinh viên được cảnh báo kết quả học tập nếu vi phạm "
                "một trong các trường hợp sau: a) Không đăng ký đủ số tín chỉ tối thiểu; "
                "b) Điểm trung bình học kỳ đạt dưới 0,8 đối với học kỳ đầu của khóa học, dưới 1,0 đối với các học kỳ tiếp theo.\n"
                "2. Sau mỗi học kỳ chính, sinh viên bị buộc thôi học do kết quả học tập yếu trong các trường hợp sau: "
                "a) Không đăng ký học hoặc không hoàn thành nghĩa vụ học phí; "
                "b) Thuộc diện cảnh báo kết quả học tập 3 lần liên tiếp; "
                "c) Vượt quá thời gian tối đa được phép học quy định tại khoản 5 Điều 2 của Quy chế này."
            ),
        },
        {
            "part": "2.6",
            "part_title": "QUY ĐỊNH VỀ CHẾ ĐỘ CHÍNH SÁCH",
            "chapter": None,
            "chapter_title": None,
            "article": 3,
            "article_title": "Điều 3. Các đối tượng được miễn, giảm học phí",
            "clauses": [1],
            "page_start": 47,
            "page_end": 47,
            "text": (
                "1. Đối tượng sinh viên được miễn 100% học phí:\n"
                "a. Các đối tượng theo quy định tại Pháp lệnh Ưu đãi người có công với cách mạng: "
                "Anh hùng Lực lượng vũ trang nhân dân, Thương binh, người hưởng chính sách như thương binh, Bệnh binh; "
                "con của liệt sỹ, con của thương binh.\n"
                "b. Sinh viên bị tàn tật, khuyết tật.\n"
                "c. Sinh viên là người dân tộc thiểu số rất ít người."
            ),
        },
        {
            "part": "2.1",
            "part_title": "QUY CHẾ ĐÀO TẠO ĐẠI HỌC",
            "chapter": "III",
            "chapter_title": "ĐÁNH GIÁ KẾT QUẢ HỌC TẬP VÀ CẤP BẰNG TỐT NGHIỆP",
            "article": 13,
            "article_title": "Điều 13. Điều kiện nhận đồ án tốt nghiệp và bảo vệ đồ án tốt nghiệp",
            "clauses": [1],
            "page_start": 24,
            "page_end": 24,
            "text": (
                "1. Sinh viên được nhận học phần đồ án tốt nghiệp (ĐATN) phải đủ các điều kiện sau tại thời điểm xét giao: "
                "a) Không bị truy cứu trách nhiệm hình sự, không trong thời gian bị kỷ luật ở mức đình chỉ học tập; "
                "b) Đạt điều kiện về kết quả học tập: đối với sinh viên chưa kết thúc khóa học: có số tín chỉ các học phần "
                "tính điểm chưa tích lũy (không bao gồm thực tập tốt nghiệp, ĐATN) không vượt quá 10 tín chỉ."
            ),
        },
    ]


def test_correct_text_with_incorrect_article_citation_is_rejected(sample_chunks) -> None:
    """Requirement 8: Answer with correct text but incorrect citation must NOT be verified."""
    # The text is about Điều 11 (cảnh báo kết quả học tập 3 lần liên tiếp bị buộc thôi học),
    # but the citation falsely attributes it to Điều 3 (Trang 47).
    hallucinated_answer = (
        "Sinh viên bị buộc thôi học nếu thuộc diện cảnh báo kết quả học tập 3 lần liên tiếp *(Khoản 1 Điều 3, Trang 47)*."
    )

    sources = _filter_supporting_sources(sample_chunks, hallucinated_answer)

    # It MUST NOT verify Điều 3 as a source because Điều 3 does not contain this rule!
    # And it MUST NOT falsely claim Điều 11 because the claim explicitly cited Điều 3.
    assert len(sources) == 0, f"Expected 0 verified sources for hallucinated citation, got: {sources}"


def test_correct_text_with_nonexistent_article_citation_is_rejected(sample_chunks) -> None:
    """Answer cites Điều 99 which does not exist in retrieved chunks."""
    hallucinated_answer = (
        "Sinh viên chưa kết thúc khóa học có số tín chỉ chưa tích lũy không vượt quá 10 tín chỉ "
        "được nhận đồ án tốt nghiệp *(Điều 99, Trang 99)*."
    )

    sources = _filter_supporting_sources(sample_chunks, hallucinated_answer)

    assert len(sources) == 0, "Non-existent cited article must not be verified against other chunks"


def test_paraphrased_answer_semantically_supported_is_verified(sample_chunks) -> None:
    """Requirement 8: Answer paraphrased from source but semantically supported must be verified."""
    # Paraphrased version of Điều 3 Khoản 1b: "Sinh viên bị tàn tật, khuyết tật" được "miễn 100% học phí"
    paraphrased_answer = (
        "Người học bị tàn tật hoặc khuyết tật thuộc đối tượng được hưởng chính sách miễn 100% học phí *(Khoản 1 Điều 3, Trang 47)*."
    )

    sources = _filter_supporting_sources(sample_chunks, paraphrased_answer)

    assert len(sources) == 1
    source = sources[0]
    assert source["article"] == 3
    assert len(source["claims"]) >= 1
    assert len(source["evidence"]) >= 1
    assert "tàn tật" in source["evidence"][0] or "khuyết tật" in source["evidence"][0]


def test_unrelated_answer_never_falls_back_to_all_chunks(sample_chunks) -> None:
    """Requirement 2: If evidence cannot be verified, never silently fall back to unrelated chunks."""
    unrelated_answer = "Nhà trường sẽ hỗ trợ xe đưa đón miễn phí cho toàn bộ sinh viên ở ký túc xá."

    sources = _filter_supporting_sources(sample_chunks, unrelated_answer)

    # In previous buggy code, `final_chunks = supported if supported else results` returned all 3 chunks!
    assert sources == [], "Must return empty sources list instead of falling back to all chunks"


def test_numerical_mismatch_is_rejected(sample_chunks) -> None:
    """If claim contains numerical value not supported by chunk (e.g. 50 tín chỉ vs 10 tín chỉ), reject."""
    incorrect_num_answer = (
        "Sinh viên chưa kết thúc khóa học có số tín chỉ chưa tích lũy không vượt quá 50 tín chỉ "
        "thì được nhận đồ án tốt nghiệp *(Điều 13, Trang 24)*."
    )

    sources = _filter_supporting_sources(sample_chunks, incorrect_num_answer)

    assert len(sources) == 0, "Numerical mismatch (50 vs 10) must cause claim verification to fail"


def test_multi_clause_answer_verifies_multiple_clauses(sample_chunks) -> None:
    """Answer covering multiple clauses in same article aggregates correctly."""
    multi_clause_answer = (
        "- Sinh viên bị cảnh báo kết quả học tập nếu điểm trung bình dưới 1,0 *(Khoản 1 Điều 11, Trang 23)*.\n"
        "- Sinh viên bị buộc thôi học nếu bị cảnh báo học tập 3 lần liên tiếp *(Khoản 2 Điều 11, Trang 23)*."
    )

    sources = _filter_supporting_sources(sample_chunks, multi_clause_answer)

    assert len(sources) == 1
    source = sources[0]
    assert source["article"] == 11
    assert len(source["claims"]) == 2
    assert len(source["evidence"]) >= 1


def test_out_of_scope_abstention_evaluation() -> None:
    """Section 8: Out of scope questions must produce empty sources and pass regression metrics."""
    refusal_answer = "Không tìm thấy tài liệu liên quan trong sổ tay sinh viên."
    response = {"answer": refusal_answer, "sources": []}

    metrics = evaluate_answer_response(
        "Trường có cấp laptop miễn phí không?",
        answerable=False,
        response=response,
    )

    assert metrics.citation_correctness
    assert metrics.faithfulness
    assert metrics.completeness
    assert metrics.scope_accuracy
    assert metrics.abstention_accuracy
