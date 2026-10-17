from app.rag.from_scratch.preprocessing import normalize_line


def test_normalize_line() -> None:
    assert normalize_line("   Sinh   viên   ") == "Sinh viên"
    assert normalize_line("Tối đa\t27 tín chỉ") == "Tối đa 27 tín chỉ"
    assert normalize_line("  ") == ""
    assert normalize_line("Điều 7. Đăng ký học tập") == "Điều 7. Đăng ký học tập"


if __name__ == "__main__":
    test_normalize_line()
    print("All tests passed!")
