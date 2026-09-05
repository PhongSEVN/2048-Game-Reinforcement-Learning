# 2048 Reinforcement Learning

Một codebase nhỏ, dễ đọc để học RL bằng cách huấn luyện agent chơi **2048**

Agent của project: **n-tuple afterstate learning**
---

## Cài đặt

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

Chỉ cần `numpy` để chạy game, baseline random, huấn luyện và chơi. `matplotlib`
dùng để vẽ `curve.png` (tùy chọn, bỏ qua nếu không cài).

---

## Bắt đầu nhanh

```bash
# 1. Tự chơi để cảm nhận game
python -m twenty48.play --size 4 --human

# 2. Baseline random (mức "không học") — ngưỡng cần vượt
python -m twenty48.play --size 4 --random --games 200 --quiet

# 3. Huấn luyện agent trên bàn nhỏ — vài phút
python -m twenty48.train_afterstate --size 3 --episodes 20000

# 4. Xem agent chơi trong giao diện đồ họa
python -m twenty48.gui --size 3 --checkpoint runs/ntuple_3x3/best.npz

# 5. Đánh giá bằng số liệu
python -m twenty48.play --size 3 --checkpoint runs/ntuple_3x3/best.npz --games 200 --quiet
```

Nên bắt đầu ở `--size 3`: không gian trạng thái nhỏ, ván ngắn, thấy điểm trung
bình vượt baseline chỉ sau vài nghìn episode. Vòng phản hồi nhanh đó chính là
lý do bắt đầu từ bàn nhỏ.

---

## Hiểu mô hình theo cách dễ nhất

Đọc phần này trước khi vào phần kiến trúc chi tiết bên dưới — nó giải thích
"mô hình" và "cách RL học" bằng ví dụ cụ thể, không công thức.

### Mô hình là gì

Không phải mạng nơ-ron. Là một **cỗ máy chấm điểm thế cờ** — nhìn 1 bàn 2048,
trả về 1 con số nói "thế này tốt cỡ nào cho tương lai". Con số đó gọi là
`V(bàn cờ)`.

Cỗ máy này không "suy luận" gì cả — nó chỉ là **nhiều bảng tra cứu cộng lại**.
Ví dụ đơn giản hóa trên bàn 3x3:

- Cắt bàn thành các mảnh nhỏ: 3 hàng, 3 cột, 4 ô vuông 2x2 → 10 mảnh.
- Với mỗi mảnh, nhìn giá trị các tile trong đó (VD hàng trên: `[4, 8, 0]`), tra
  vào 1 bảng số ứng với đúng mảnh đó, lấy ra 1 số.
- `V(bàn) = tổng của 10 số tra được từ 10 mảnh.`

Ban đầu mọi con số trong mọi bảng đều là **0**. Cỗ máy chưa biết gì cả. Nó học
bằng cách chỉnh dần các con số này qua hàng chục nghìn ván chơi thật — đây
chính là phần "reinforcement learning".

### Agent chọn nước đi thế nào

2048 đặc biệt ở chỗ: trượt bàn sang 1 hướng là **hoàn toàn đoán trước được**
(không có yếu tố ngẫu nhiên) — chỉ có việc *sinh tile mới sau đó* mới ngẫu
nhiên. Nên trước khi đi, agent thử cả 4 hướng trong đầu:

```
hướng LÊN    → giả lập trượt → được X điểm gộp → bàn sau khi trượt: A → điểm = X + V(A)
hướng XUỐNG  → giả lập trượt → được Y điểm gộp → bàn sau khi trượt: B → điểm = Y + V(B)
hướng TRÁI   → ...                                                   → điểm = ...
hướng PHẢI   → ...                                                   → điểm = ...
```

Chọn hướng có điểm cao nhất. Đơn giản vậy thôi — không tìm kiếm sâu, chỉ nhìn
1 bước rồi hỏi "cỗ máy chấm điểm nghĩ bàn sau nước này tốt cỡ nào".

Bàn cờ *ngay sau khi trượt, trước khi sinh tile mới* gọi là **afterstate** —
thuật ngữ này xuất hiện xuyên suốt phần còn lại của README và trong code
(`AfterstateAgent`, `V(afterstate)`).

### RL nằm ở đâu — nó học ra sao

Không ai dán nhãn "nước này đúng, nước kia sai". Agent tự chơi, rồi **tự so
sánh dự đoán của chính mình với thực tế xảy ra sau đó**, lệch bao nhiêu thì
chỉnh bảng số bấy nhiêu.

Ví dụ cụ thể, một bước:

1. Agent vừa đi 1 nước, bàn cờ sau khi trượt (afterstate) là bàn `A`. Lúc
   quyết định đi nước này, cỗ máy đã chấm `V(A) = 100`.
2. Trò chơi sinh tile mới, agent đi tiếp nước kế tiếp: được thêm `8` điểm gộp,
   afterstate của nước đó là `A'`, được chấm `V(A') = 95`.
3. Câu hỏi: dự đoán `V(A) = 100` có khớp thực tế không? "Thực tế" ước lượng
   được = điểm vừa kiếm thêm + giá trị bàn kế tiếp = `8 + 95 = 103`.
4. Lệch = `103 − 100 = 3` (dự đoán hơi thấp). Cỗ máy **nhích các con số trong
   10 bảng đã dùng để tính `V(A)` lên một chút** — theo hướng làm `V(A)` gần
   103 hơn.
5. Lặp lại y hệt vậy ở mọi nước, của mọi ván, hàng chục nghìn lần.

Đây chính là ý tưởng **temporal-difference learning**: thay vì đợi hết ván mới
biết đúng sai (quá chậm, quá loãng), agent dùng ngay dự đoán của bước *kế
tiếp* làm "đáp án tạm" để chỉnh dự đoán ở bước *hiện tại*.

### Vì sao lặp lại nhiều thì agent giỏi lên

Những mẫu bàn cờ hay dẫn tới điểm cao về sau (nhiều ô trống, tile lớn dồn về
góc, hàng dễ gộp tiếp) — mỗi lần xuất hiện đều được nhích điểm lên. Mẫu dẫn
tới bí nước, phải gộp lung tung — bị nhích điểm xuống. Sau đủ nhiều ván, các
bảng số "học" được: mẫu nào tốt, mẫu nào xấu. Agent chọn nước dựa vào các bảng
đó → tự nhiên né mẫu xấu, tìm mẫu tốt, dù chưa ai dạy nó khái niệm "chiến
thuật" nào cả.

Phần "Kiến trúc model chi tiết" bên dưới viết lại đúng ví dụ này bằng công
thức và code thật (`ntuple.py`) — đọc phần này trước để không bị choáng bởi
ký hiệu.

---

## Giao diện đồ họa (`gui.py`)

Cửa sổ Tkinter — chỉ dùng thư viện chuẩn, không cần cài thêm.

```bash
# agent tự chơi
python -m twenty48.gui --size 3 --checkpoint runs/ntuple_3x3/best.npz

# baseline random
python -m twenty48.gui --size 3 --random

# tự chơi bằng phím mũi tên
python -m twenty48.gui --size 4 --human
```

**Điều khiển:**

| Nút / phím | Tác dụng |
|---|---|
| Play / Pause | bật/tắt tự chơi (chế độ agent hoặc random) |
| Step | đi một nước |
| New Game | ván mới |
| thanh trượt `ms / move` | tốc độ tự chơi |
| radio `agent / random / human` | đổi chế độ |
| phím mũi tên | đi nước trong chế độ human |

**Hành vi khi thua:** mặc định dừng lại, canvas hiện `GAME OVER`, nút về
`Play`. Bấm `Play` hoặc `Step` lúc đó sẽ tự tạo ván mới. Muốn tự chạy liên tục
thì thêm cờ `--autorestart`.

Bàn cờ tự co giãn để vừa màn hình; cỡ chữ trong ô tự thu nhỏ khi số dài.

---

## Bản đồ: khái niệm RL ↔ vị trí trong code

Bảng tra nhanh, sau khi đã đọc phần giải thích trực quan ở trên:

| Khái niệm RL | Nghĩa trong 2048 | File / ký hiệu |
|---|---|---|
| **Environment** | game + việc sinh tile ngẫu nhiên | `env.py` `Env2048` |
| **State `s`** | bàn cờ, giá trị tile thô `int32` shape `(N, N)` | `Env2048.board` |
| **Action `a`** | lên / xuống / trái / phải | `board.py` `ACTIONS` |
| **Afterstate** | bàn cờ *ngay sau khi trượt/gộp, trước khi sinh tile* | `board.step_move` |
| **Transition** | trượt, gộp, sinh tile 2/4 ngẫu nhiên | `board.step_move` + `board.spawn_tile` |
| **Value `V(afterstate)`** | tổng các bảng tra n-tuple | `ntuple.py` `NTupleNetwork.value` |
| **Policy `π(s)`** | tìm kiếm 1 bước: `argmax [reward + V(afterstate)]` trên nước **hợp lệ** | `ntuple.py` `AfterstateAgent.best` |
| **Bootstrapping / TD(0)** | `target = r_kế_tiếp + V(afterstate_kế_tiếp)` | `ntuple.py` `train_episode` |
| **Episode** | một ván, kết thúc khi hết nước hợp lệ | `board.is_game_over` |

### Mẹo riêng cho 2048: mask nước hợp lệ

Một nước không làm bàn thay đổi là nước không hợp lệ. `AfterstateAgent.best`
chỉ xét các nước làm `board.step_move` trả về `changed=True`. Không có mẹo
này, agent phí thời gian đầu để học "đừng chọn nước vô nghĩa", rồi phải học
lại điều đó cho từng bàn.

---

## Kiến trúc model chi tiết

Bản chính xác, bằng công thức, của ví dụ ở phần "Hiểu mô hình theo cách dễ
nhất" — cùng một ý tưởng, giờ khớp 1-1 với code trong `ntuple.py`.

### 1. N-tuple network — `NTupleNetwork` trong `ntuple.py`

Không phải mạng nơ-ron. Là **tổng của nhiều bảng tra cứu** = một mô hình tuyến
tính thưa khổng lồ trên các mẫu cục bộ của bàn cờ. Không có phi tuyến, không có
bias ngoài chính các ô trong bảng.

**Thành phần:**

- `templates` — danh sách các nhóm ô ("mảnh", như ví dụ ở trên). 3x3: 3 hàng +
  3 cột (mỗi cái 3 ô) + 4 ô vuông 2x2 (mỗi cái 4 ô) = **10 template**. (Xem
  bảng theo size bên dưới.)
- `tables[t]` — một mảng `float64` kích thước `max_exp ^ len(template_t)`, khởi
  tạo bằng 0. Mỗi ô trong mảng = một trọng số học được cho **một mẫu cục bộ cụ
  thể**.

| Kích thước | Tuple dùng | Ghi chú |
|---|---|---|
| N ≤ 5 | mỗi hàng + mỗi cột + mỗi ô 2x2 | phủ toàn bộ, bảng nhỏ |
| 6 ≤ N ≤ 7 | mọi triple ngang/dọc + ô 2x2 | giữ tuple ngắn để bảng không nổ (`max_exp ^ len`) |
| 8 ≤ N ≤ 10 | chỉ ô 2x2 | 241 tuple trên bàn 10x10 làm mỗi lần `value()` quá chậm; ~81 ô 2x2 vẫn phủ mọi ô |

| Bàn | Template | Tổng trọng số |
|---|---|---|
| 3x3 | 6 × 15³ + 4 × 15⁴ | 222,750 |
| 4x4 | 17 × 15⁴ | 860,625 |
| 10x10 | 81 × 15⁴ | 4,100,625 |

**Đánh chỉ số** — với template `t` gồm các ô `c₀…c_{k-1}` và số mũ tile tương
ứng `e₀…e_{k-1}` (mỗi `eᵢ ∈ [0, max_exp-1]`, ví dụ tile 8 → số mũ 3 vì `2³=8`):

```
idx_t = e₀·max_exp⁰ + e₁·max_exp¹ + … + e_{k-1}·max_exp^{k-1}
```

Đây là mã vị trí cơ số `max_exp` của véc-tơ số mũ trên các ô của template — một
song ánh giữa "mẫu cục bộ" và "một hàng trong bảng". Đây chính là "tra vào 1
bảng số ứng với đúng mảnh đó" ở ví dụ trực quan phía trên.

**Hàm giá trị:**

```
V(board) = Σ_{t=0}^{n-1}  tables[t][ idx_t(board) ]
```

**Góc nhìn tuyến tính:** đặt `φ(board)` là véc-tơ nhị phân dài
`Σ_t max_exp^{k_t}`, có đúng `n` bit bằng 1 (mỗi bảng một bit). Khi đó
`V = wᵀ φ`, với `w` là toàn bộ các bảng nối lại. Cập nhật bên dưới chính là
TD tuyến tính chuẩn `w ← w + α·δ·φ`.

**Cập nhật** (`NTupleNetwork.update(board, δ, lr)`):

```
step = lr · δ / n_template
với mỗi template t:  tables[t][ idx_t(board) ] += step
```

Chia `lr` cho số template để độ lớn bước ổn định bất kể bàn to nhỏ. Đây chính
là bước 4 trong ví dụ trực quan ("nhích các con số trong 10 bảng lên một
chút").

### 2. Policy afterstate — `AfterstateAgent.best`

Nước đi 2048 tất định; chỉ tile sinh ra sau đó mới ngẫu nhiên. Nên tìm kiếm
**1 bước** trên "điểm gộp tức thì + giá trị bàn sau khi trượt":

```
với mỗi nước hợp lệ a:
    after, gained, changed = board.step_move(s, a)   # trượt + gộp, KHÔNG sinh tile
    nếu không changed: bỏ qua
    score(a) = gained + V(after)
a*      = argmax_a score(a)
trả về (a*, after của a*, gained của a*)
```

Với xác suất `ε` (mặc định 0): chọn `a` ngẫu nhiên trong nước hợp lệ.

### 3. Vòng huấn luyện — TD(0) trên afterstate (`train_episode`)

```
s ← env.reset()
(a, w, g) ← best(s)                 # w = afterstate, g = điểm nước a

lặp khi a ≠ None:
    s', terminated, truncated ← env.step(a)      # <-- sinh tile ngẫu nhiên ở đây
    nếu truncated và không terminated:
        break                                    # bị --max-steps cắt, KHÔNG cập nhật
    nếu terminated:
        V(w) ← V(w) + lr·(0 − V(w))              # trạng thái cuối, target = 0
        break
    (a', w', g') ← best(s')
    nếu a' == None:
        V(w) ← V(w) + lr·(0 − V(w)); break
    r  ← g' + empty_bonus · (số ô trống trong s')
    δ  ← r + V(w') − V(w)                        # bootstrap: reward nước KẾ + V(afterstate kế)
    V(w) ← V(w) + lr·δ
    (a, w, g) ← (a', w', g')
```

- Target cho `V(afterstate_t)` = **điểm của nước kế tiếp** cộng
  `V(afterstate_{t+1})`. Không có `γ` (coi như `γ = 1`) — ván hữu hạn.
- Reward ở đây là **điểm gộp thô** `g'` (cộng shaping nếu bật) — `V` có thang
  bằng "tổng điểm kỳ vọng còn lại tới hết ván".
- Cập nhật online, ngay trong lúc chơi, mỗi nước một lần — đúng ví dụ 5 bước ở
  phần trực quan phía trên, viết lại bằng ký hiệu.

### 4. Sinh tile của môi trường (`board.spawn_tile`)

Sau mỗi nước làm bàn thay đổi: chọn **đều** một ô trống, đặt tile **2 với xác
suất 0.9**, **4 với xác suất 0.1**.

---

## Kết quả tham khảo (đánh giá greedy)

### 3x3 — thử thách là *sống sót*

| Agent | Điểm trung bình | Tile lớn nhất thường gặp |
|---|---|---|
| random | ~180 | 32 |
| n-tuple afterstate, 20k episode | ~1950 | 256 (thỉnh thoảng 512) |

### 4x4 — n-tuple học cực nhanh per-episode

Dừng ở ~900 episode (~13 phút CPU):

| metric | giá trị |
|---|---|
| điểm trung bình | ~11,600 |
| tile trung bình | ~800 |
| tile lớn nhất đạt được | 2048 |

### 10x10 — thử thách đổi bản chất

Trên bàn 100 ô, agent **gần như không bao giờ thua** (deadlock cực hiếm). "Điểm"
không còn đo sống-sót mà đo *gộp được bao nhiêu trong số nước cho phép*. Huấn
luyện cap ở 500 nước/ván cho nhanh, nên đường cong phẳng; thả cap ra thì cùng
agent đó lên cao hơn:

| | train (cap 500 nước) | eval (cap 1500 nước) |
|---|---|---|
| điểm trung bình | ~6,700 | ~25,900 |
| tile trung bình | 512 | ~1,430 |
| tile lớn nhất | 1024 | 2048 |
| chết trước cap | 0% | 0/20 |

---

## Danh sách file

```
twenty48/
  board.py             logic game thuần cho bàn N x N (trượt, gộp, sinh tile, render)
  env.py               environment RL: reset / step / observe / legal_actions
  ntuple.py            NTupleNetwork + AfterstateAgent  (afterstate TD)
  train_afterstate.py  vòng huấn luyện n-tuple + CLI
  play.py              xem agent / random / human + thống kê tổng hợp
  gui.py               giao diện Tkinter: agent tự chơi, hoặc tự chơi
```

Sản phẩm huấn luyện nằm dưới `runs/`:

```
runs/ntuple_3x3/
  best.npz     trọng số có điểm moving-average tốt nhất
  last.npz     trọng số gần nhất
  scores.csv   episode,score,max_tile,steps
  curve.png    đường cong điểm moving-average (nếu có matplotlib)
```
