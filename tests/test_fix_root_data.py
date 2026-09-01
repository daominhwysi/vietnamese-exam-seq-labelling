import unittest
from src.data.fix_root_data import fix_xml_sections, fix_xml_stimulus_citations, fix_json_stimulus

class TestFixRootData(unittest.TestCase):
    def test_untag_exam_metadata(self):
        input_xml = (
            "<section># SỞ GD&ĐT BẮC NINH</section>\n"
            "<section>## TRƯỜNG THPT CHUYÊN BẮC NINH</section>\n"
            "<section># ĐỀ THI THỬ TỐT NGHIỆP THPT</section>\n"
            "<section>**MÃ ĐỀ: 101**</section>\n"
            "<section>Thời gian làm bài: 50 phút</section>\n"
            "<section>Họ và tên thí sinh: Nguyễn Văn A</section>\n"
            "<section>## PHẦN I. Câu trắc nghiệm nhiều phương án lựa chọn</section>\n"
            "<question_label>Câu 1:</question_label> <stem>Nội dung câu 1</stem>"
        )
        expected_xml = (
            "# SỞ GD&ĐT BẮC NINH\n"
            "## TRƯỜNG THPT CHUYÊN BẮC NINH\n"
            "# ĐỀ THI THỬ TỐT NGHIỆP THPT\n"
            "**MÃ ĐỀ: 101**\n"
            "Thời gian làm bài: 50 phút\n"
            "Họ và tên thí sinh: Nguyễn Văn A\n"
            "<section>## PHẦN I. Câu trắc nghiệm nhiều phương án lựa chọn</section>\n"
            "<question_label>Câu 1:</question_label> <stem>Nội dung câu 1</stem>"
        )
        fixed = fix_xml_sections(input_xml)
        self.assertEqual(fixed, expected_xml)

    def test_untag_footers_and_noise(self):
        input_xml = (
            "<question_label>Câu 10:</question_label> <stem>Nội dung</stem>\n"
            "<section>----- HẾT -----</section>\n"
            "<section>Thí sinh không được sử dụng tài liệu.</section>\n"
            "<section>###</section>\n"
            "<section>hoặc</section>"
        )
        expected_xml = (
            "<question_label>Câu 10:</question_label> <stem>Nội dung</stem>\n"
            "----- HẾT -----\n"
            "Thí sinh không được sử dụng tài liệu.\n"
            "###\n"
            "hoặc"
        )
        fixed = fix_xml_sections(input_xml)
        self.assertEqual(fixed, expected_xml)

    def test_convert_misclassified_question_labels(self):
        input_xml = (
            "<section>## Câu 4. (6,0 điểm)</section>\n"
            "<stem>Cho hình chóp S.ABCD...</stem>\n"
            "<section>## Question 15</section>\n"
            "<stem>Fill in the blank...</stem>\n"
            "<section>### Bài 1 (2,0 điểm):</section>\n"
            "<stem>Giải phương trình</stem>"
        )
        expected_xml = (
            "<question_label>## Câu 4. (6,0 điểm)</question_label>\n"
            "<stem>Cho hình chóp S.ABCD...</stem>\n"
            "<question_label>## Question 15</question_label>\n"
            "<stem>Fill in the blank...</stem>\n"
            "<question_label>### Bài 1 (2,0 điểm):</question_label>\n"
            "<stem>Giải phương trình</stem>"
        )
        fixed = fix_xml_sections(input_xml)
        self.assertEqual(fixed, expected_xml)

    def test_convert_solutions_and_barems(self):
        input_xml = (
            "<section># HƯỚNG DẪN GIẢI CHI TIẾT</section>\n"
            "<section>## ĐÁP ÁN VÀ LỜI GIẢI</section>\n"
            "<section>**Lời giải**</section>\n"
            "<section><table><tr><td>1. A</td><td>2. B</td></tr></table></section>"
        )
        expected_xml = (
            "<explanation># HƯỚNG DẪN GIẢI CHI TIẾT</explanation>\n"
            "<explanation>## ĐÁP ÁN VÀ LỜI GIẢI</explanation>\n"
            "<explanation>**Lời giải**</explanation>\n"
            "<explanation><table><tr><td>1. A</td><td>2. B</td></tr></table></explanation>"
        )
        fixed = fix_xml_sections(input_xml)
        self.assertEqual(fixed, expected_xml)

    def test_preserve_true_sections_and_directions(self):
        input_xml = (
            "<section># PHẦN II. Câu trắc nghiệm đúng sai\n\nThí sinh trả lời từ câu 1 đến câu 4.</section>\n"
            "<section>Part 1: Incomplete Sentences</section>\n"
            "<section>## Chủ đề Địa lí có 17 câu hỏi từ 501 đến 517</section>\n"
            "<section># TƯ DUY TOÁN HỌC</section>\n"
            "<section>## Dạng 1: Tìm tập xác định</section>\n"
            "<section>Mark the letter A, B, C, or D on your answer sheet to indicate the correct answer</section>"
        )
        fixed = fix_xml_sections(input_xml)
        self.assertEqual(fixed, input_xml)

    def test_unwrap_nested_section_inside_stimulus(self):
        input_xml = (
            "<stimulus>Read the following leaflet:\n"
            "<section>## MINDCARE FOR YOUTH</section>\n"
            "Mental wellness is vital for teens.\n"
            "(Adapted from Health Mag)</stimulus>"
        )
        expected_xml = (
            "<stimulus>Read the following leaflet:\n"
            "## MINDCARE FOR YOUTH\n"
            "Mental wellness is vital for teens.\n"
            "(Adapted from Health Mag)</stimulus>"
        )
        fixed = fix_xml_sections(input_xml)
        self.assertEqual(fixed, expected_xml)

if __name__ == "__main__":
    unittest.main()
