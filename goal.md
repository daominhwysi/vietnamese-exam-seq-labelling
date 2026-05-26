Tôi đang làm một corpus để giả lập đề thi ở Việt Nam bằng AI
Có một số tiêu chí ngẫu nhiên sẽ được random bằng thuật toán để build prompt:
Môn học: ['Kinh tế pháp luật', 'Địa lý', 'Lịch sử', 'Toán Đại số', 'Toán hình học', 'Vật lý' , 'Hóa học']
Lớp: [8,9,10,11,12]
Dạng câu hỏi: [Trắc nghiệm nhiều phương án, Đúng sai, Trả lời ngắn]
Độ dài các phần:

- Trắc nghiệm Random với trung bình stem là 30 từ
- Đúng sai trung bình 100 từ
- Trả lời trung bình 150 từ

Các dạng câu hỏi đặc biệt ở phần trả lời ngắn
Từ một thông tin trả lời 2-3 câu trả lời ngắn

Các dạng trắc nghiệm đặc biệt
Từ một thông tin trả lời từ 3-4 câu trắc nghiệm

- Các dạng đặc biệt là các dạng hiếm có tỉ lệ xuất hiện thấp hơn
- Yêu cầu AI làm ngẫu nhiên, độc lạ nhất có thể, tránh trùng lặp

## Example

Trắc nghiệm:
Câu 5: Họ nguyên hàm của hàm số $f(x) = x^2$ là:

A. $\frac{1}{3}x^3 + C.$

B. $2x^3 + C.$

C. $3x^3 + C.$

D. $\frac{1}{2}x^3 + C.$

Đúng sai:

**Câu 3:** Mô hình toán học sau đây được sử dụng trong quan sát chuyển động của một vật. Trong không gian cho hệ tọa độ $Oxyz$ có $\vec{i}, \vec{j}, \vec{k}$ lần lượt là các vectơ đơn vị trên các trục $Ox, Oy, Oz$ và độ dài của mỗi vectơ đơn vị đó bằng 1 mét. Cho hai điểm $A$ và $B$, trong đó điểm $A$ có tọa độ là $(5;5;0)$. Một vật (coi như là một hạt) chuyển động thẳng với tốc độ phụ thuộc thời gian $t$ (giây) theo công thức $v(t) = \beta t + 300$ (m/giây), trong đó $\beta$ là hằng số dương và $0 \le t \le 6$. Ở thời điểm ban đầu ($t = 0$), vật đi qua $A$ với tốc độ $300\text{ m/giây}$ và hướng tới $B$. Sau 2 giây kể từ thời điểm ban đầu, vật đi được quãng đường $604\text{ m}$. Gọi $\vec{u} = (a;b;c)$ là vectơ cùng hướng với vectơ $\vec{AB}$. Biết rằng $|\vec{u}| = 1$ và góc giữa vectơ $\vec{u}$ lần lượt với các vectơ $\vec{i}, \vec{j}, \vec{k}$ có số đo tương ứng bằng $60^\circ, 60^\circ, 45^\circ$.

**a)** $a = \cos 60^\circ$.

**b)** Phương trình đường thẳng $AB$ là $\frac{x-5}{1} = \frac{y-5}{1} = \frac{z}{2}$.

**c)** $\beta = 2$.

**d)** Giả sử sau 5 giây kể từ thời điểm ban đầu, vật đến điểm $B(x_B; y_B; z_B)$. Khi đó $x_B > 768$.

Trả lời ngắn:

**Câu 2:** Nếu một doanh nghiệp sản xuất $x$ sản phẩm trong một tháng $(x \in \mathbb{N}^*; 1 \le x \le 4\,500)$ thì doanh thu nhận được khi bán hết số sản phẩm đó là $F(x) = -0,01x^2 + 300x$ (nghìn đồng), trong khi chi phí sản xuất bình quân cho mỗi sản phẩm là $G(x) = \frac{30\,000}{x} + 200$ (nghìn đồng). Giả sử số sản phẩm sản xuất ra luôn được bán hết. Trong một tháng, doanh nghiệp đó cần sản xuất ít nhất bao nhiêu sản phẩm để lợi nhuận thu được lớn hơn 100 triệu đồng?

## Output

Hãy yêu cầu AI xuất ra XML rồi parse về json cho dễ, các công thức toán dùng latex bọc trong $....$
Ví dụ output:
<question>
<stem>Họ nguyên hàm của hàm số $f(x) = x^2$ là</stem>

<option>$\frac{1}{3}x^3 + C.$</option>

<option>$2x^3 + C.$
</option>
<option>$3x^3 + C.$
</option>
<option>$\frac{1}{2}x^3 + C.$
</option>
</question>
Với câu hỏi nhóm:
<group_question>
  <context>Đoạn thông tin dùng chung cho câu 1 và câu 2...</context>

  <question>
    <stem>Câu hỏi 1...</stem>
    <option>...</option>
    ...
  </question>

  <question>
    <stem>Câu hỏi 2...</stem>
    <option>...</option>
    ...
  </question>
</group_question>
