import tensorflow as tf
import numpy as np
import threading
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle



ROWS = 100
COLS = 100
NUM_THREADS = 4

np.random.seed(10)


A_tf = tf.constant(
    np.random.randint(1, 10, (ROWS, COLS)),
    dtype=tf.int32
)

B_tf = tf.constant(
    np.random.randint(1, 10, (ROWS, COLS)),
    dtype=tf.int32
)

A = A_tf.numpy()
B = B_tf.numpy()

C = np.zeros((ROWS, COLS), dtype=np.int64)


def multiply(start_row, end_row, thread_id):

    for i in range(start_row, end_row):
        for j in range(COLS):

            total = 0

            for k in range(COLS):
                total += int(A[i][k]) * int(B[k][j])

            C[i][j] = total


threads = []
rows_per_thread = ROWS // NUM_THREADS

for t in range(NUM_THREADS):

    start = t * rows_per_thread

    if t == NUM_THREADS - 1:
        end = ROWS
    else:
        end = start + rows_per_thread

    th = threading.Thread(
        target=multiply,
        args=(start, end, t),
        name=f"Thread-{t + 1}"
    )

    threads.append(th)
    th.start()

for th in threads:
    th.join()



tf_result = tf.matmul(A_tf, B_tf).numpy()

print("\n==============================================")
print("       THREADED MATRIX MULTIPLICATION")
print("==============================================")
print("Matrix A       : 100 x 100")
print("Matrix B       : 100 x 100")
print("Number Threads : 4")

if np.array_equal(C, tf_result):
    print("TensorFlow verification : PASSED")
else:
    print("TensorFlow verification : FAILED")

print("==============================================")

print("\nSample Thread Operations:")

samples = [
    (0, 0, 0),
    (25, 0, 0),
    (50, 0, 0),
    (75, 0, 0),
    (0, 1, 1),
    (25, 1, 1),
    (50, 1, 1),
    (75, 1, 1)
]

for i, j, k in samples:

    thread_no = (i // rows_per_thread) + 1

    a_value = int(A[i][k])
    b_value = int(B[k][j])
    product = a_value * b_value

    print(
        f"Thread-{thread_no}: "
        f"A[{i}][{k}] x B[{k}][{j}] = "
        f"{a_value} x {b_value} = {product}"
    )




fig, ax = plt.subplots(figsize=(16, 9))

try:
    fig.canvas.manager.set_window_title(
        "Threaded Matrix Multiplication"
    )
except Exception:
    pass

# Try to maximize the Matplotlib window.
try:
    manager = plt.get_current_fig_manager()
    manager.window.state("zoomed")
except Exception:
    try:
        manager = plt.get_current_fig_manager()
        manager.window.showMaximized()
    except Exception:
        pass

ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")



A_x, A_y = 3, 20
B_x, B_y = 37.5, 20
C_x, C_y = 72, 20

SIZE = 25
CELL = SIZE / 100


def draw_grid(x, y):

    ax.add_patch(
        Rectangle(
            (x, y),
            SIZE,
            SIZE,
            fill=False,
            linewidth=1.5
        )
    )

    
    for n in range(1, 100):

        p = n * CELL

        ax.plot(
            [x + p, x + p],
            [y, y + SIZE],
            linewidth=0.10
        )

        ax.plot(
            [x, x + SIZE],
            [y + p, y + p],
            linewidth=0.10
        )


draw_grid(A_x, A_y)
draw_grid(B_x, B_y)
draw_grid(C_x, C_y)



ax.text(
    50, 53,
    "THREADED MATRIX MULTIPLICATION",
    ha="center",
    fontsize=21,
    fontweight="bold"
)

ax.text(
    A_x + SIZE / 2, 48,
    "Matrix A (100 × 100)",
    ha="center",
    fontsize=13,
    fontweight="bold"
)

ax.text(
    B_x + SIZE / 2, 48,
    "Matrix B (100 × 100)",
    ha="center",
    fontsize=13,
    fontweight="bold"
)

ax.text(
    C_x + SIZE / 2, 48,
    "Result C (100 × 100)",
    ha="center",
    fontsize=13,
    fontweight="bold"
)

ax.text(
    34.5, 31,
    "×",
    ha="center",
    fontsize=32,
    fontweight="bold"
)

ax.text(
    69.5, 31,
    "=",
    ha="center",
    fontsize=32,
    fontweight="bold"
)



MARKER_CELLS = 3
MARKER = CELL * MARKER_CELLS

marker_A = Rectangle(
    (A_x, A_y + SIZE - MARKER),
    MARKER,
    MARKER,
    fill=False,
    linewidth=3
)

marker_B = Rectangle(
    (B_x, B_y + SIZE - MARKER),
    MARKER,
    MARKER,
    fill=False,
    linewidth=3
)

marker_C = Rectangle(
    (C_x, C_y + SIZE - MARKER),
    MARKER,
    MARKER,
    fill=False,
    linewidth=3
)

ax.add_patch(marker_A)
ax.add_patch(marker_B)
ax.add_patch(marker_C)



info = ax.text(
    50, 10,
    "",
    ha="center",
    fontsize=12
)

progress = ax.text(
    50, 6,
    "",
    ha="center",
    fontsize=10
)



PATH_LENGTH = 100 * 100

def row_col(n):
    """Row-major position: visits every matrix cell."""
    return n // 100, n % 100


def col_row(n):
    """Column-major position: visits every matrix cell."""
    return n % 100, n // 100



STEP = 5

TOTAL_FRAMES = PATH_LENGTH // STEP




def cell_position(x, y, row, col):

    px = x + col * CELL
    py = y + (99 - row) * CELL

    return px, py


def smooth(a, b, amount):

    
    s = amount * amount * (3 - 2 * amount)
    return a + (b - a) * s




def update(frame):

    n = (frame * STEP) % PATH_LENGTH

    # Current positions
    a_row, a_col = row_col(n)
    b_row, b_col = col_row(n)

    # Result follows a different diagonal-like scan.
    c_row = (n // 100) % 100
    c_col = (n + (n // 100) * 3) % 100

   
    ax_pos, ay_pos = cell_position(
        A_x, A_y, a_row, a_col
    )

    marker_A.set_xy(
        (ax_pos, ay_pos)
    )

    
    bx_pos, by_pos = cell_position(
        B_x, B_y, b_row, b_col
    )

    marker_B.set_xy(
        (bx_pos, by_pos)
    )

    
    cx_pos, cy_pos = cell_position(
        C_x, C_y, c_row, c_col
    )

    marker_C.set_xy(
        (cx_pos, cy_pos)
    )

    
    i = a_row
    k = a_col
    j = b_col

    a_value = int(A[i][k])
    b_value = int(B[k][j])
    product = a_value * b_value

    thread_no = (i // rows_per_thread) + 1

    # Partial sum for display only.
    partial = int(
        np.dot(A[i, :k + 1], B[:k + 1, j])
    )

    info.set_text(
        f"Thread-{thread_no}   |   "
        f"A[{i}][{k}] × B[{k}][{j}]   |   "
        f"{a_value} × {b_value} = {product}   |   "
        f"C[{i}][{j}] = {partial}"
    )

    progress.set_text(
        f"Processing matrix cells: "
        f"{a_row + 1}/100 × {a_col + 1}/100"
    )

    return (
        marker_A,
        marker_B,
        marker_C,
        info,
        progress
    )





print("\nCreating GIF... Please wait.")

gif_ani = FuncAnimation(
    fig,
    update,
    frames=200,
    interval=25,
    repeat=False,
    blit=False
)

gif_ani.save(
    "matrix_multiplication.gif",
    writer="pillow",
    fps=15
)

print("GIF created successfully!")

plt.show()
