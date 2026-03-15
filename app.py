# app.py

from flask import Flask, render_template, request, jsonify
import numpy as np
import pulp
import os
import sys
import webbrowser
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

app = Flask(__name__)

def get_solver_path():
    if hasattr(sys, '_MEIPASS'):
        # Внутри .exe файл будет лежать в корне временной папки
        return os.path.join(sys._MEIPASS, "cbc.exe")
    # Путь для запуска из IDE (укажите тот, который нашли)
    return r"D:\Python projects\SZI\.venv\Lib\site-packages\pulp\solverdir\cbc\win\i64\cbc.exe"

def load_security_data(file_path):

    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    m, n = map(int, lines[0].split())

    matrix = []
    for i in range(1, m + 1):
        matrix.append(list(map(float, lines[i].split())))
    A = np.array(matrix)

    line_idx = m + 1
    b = np.array(list(map(float, lines[line_idx].split())))

    line_idx += 1
    c = np.array(list(map(float, lines[line_idx].split())))

    line_idx += 1
    d = np.array(list(map(float, lines[line_idx].split())))

    line_idx += 1
    lam = float(lines[line_idx])

    # сортировка d по убыванию
    indices = np.argsort(d)[::-1]

    return {
        "m": m,
        "n": n,
        "A": A[:, indices],
        "b": b,
        "c": c[indices],
        "d": d[indices],
        "lambda": lam,
        "original_indices": indices
    }


def solve_knapsack_pulp(c, A, b):

    n = len(c)
    m = len(A)

    prob = pulp.LpProblem("Knapsack", pulp.LpMaximize)

    x = [pulp.LpVariable(f"x_{j}", cat=pulp.LpBinary) for j in range(n)]

    prob += pulp.lpSum(c[j] * x[j] for j in range(n))

    for i in range(m):
        prob += pulp.lpSum(A[i][j] * x[j] for j in range(n)) <= b[i]

    solver = pulp.COIN_CMD(path=get_solver_path(), msg=0)
    prob.solve(solver)

    if pulp.LpStatus[prob.status] == "Optimal":
        return [int(pulp.value(x[j])) for j in range(n)]

    return None


def step2(x0):

    indices = [i for i, v in enumerate(x0) if v == 1]

    if not indices:
        return 0

    return max(indices) + 1


def solve_subproblem(args):

    data, j0, s = args

    n = data["n"]
    m = data["m"]
    c = data["c"]
    d = data["d"]
    A = data["A"]
    b = data["b"]
    lam = data["lambda"]

    num_vars = j0 - s

    prob = pulp.LpProblem("sub", pulp.LpMaximize)

    x = [pulp.LpVariable(f"x_{j}", cat=pulp.LpBinary) for j in range(num_vars)]

    objective = pulp.lpSum(lam * c[j] * x[j] for j in range(num_vars - 1))

    last = num_vars - 1
    objective += (lam * c[last] + (1 - lam) * d[last]) * x[last]

    prob += objective

    for i in range(m):
        prob += pulp.lpSum(A[i][j] * x[j] for j in range(num_vars)) <= b[i]

    solver = pulp.COIN_CMD(path=get_solver_path(), msg=0)
    prob.solve(solver)

    if pulp.LpStatus[prob.status] == "Optimal":

        xs = [int(pulp.value(x[j])) for j in range(num_vars)]

        return xs + [0] * (n - num_vars)

    return None


def solve_problem(data):

    x0 = solve_knapsack_pulp(data["c"], data["A"], data["b"])

    if x0 is None:
        return None

    V = [x0]

    j0 = step2(x0)

    tasks = [(data, j0, s) for s in range(1, j0)]

    with ProcessPoolExecutor() as executor:
        results = executor.map(solve_subproblem, tasks)

    for r in results:
        if r:
            V.append(r)

    lam = data["lambda"]

    best_F = -1
    best_x = None

    for x in V:

        F1 = sum(data["c"][j] * x[j] for j in range(len(x)))

        selected_d = [data["d"][j] for j in range(len(x)) if x[j] == 1]
        F2 = min(selected_d) if selected_d else 0

        F = lam * F1 + (1 - lam) * F2

        if F > best_F:
            best_F = F
            best_x = x

    original = data["original_indices"]

    selected = [int(original[i] + 1) for i, v in enumerate(best_x) if v == 1]

    selected.sort()

    # map best_x back to original ordering
    n = data["n"]
    x_orig = [0] * n
    for sorted_pos, val in enumerate(best_x):
        orig_pos = int(original[sorted_pos])
        x_orig[orig_pos] = val

    print(f"Optimal x vector: {x_orig}")

    return best_F, selected, x_orig


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():

    file = request.files["file"]

    path = "temp.txt"
    file.save(path)

    data = load_security_data(path)

    return jsonify({
        "m": data["m"],
        "n": data["n"],
        "lambda": data["lambda"],
        "A": data["A"].tolist(),
        "b": data["b"].tolist(),
        "c": data["c"].tolist(),
        "d": data["d"].tolist()
    })


@app.route("/solve", methods=["POST"])
def solve():

    data = request.json

    dataset = {
        "m": int(data["m"]),
        "n": int(data["n"]),
        "A": np.array(data["A"]),
        "b": np.array(data["b"]),
        "c": np.array(data["c"]),
        "d": np.array(data["d"]),
        "lambda": float(data["lambda"]),
        "original_indices": np.arange(len(data["c"]))
    }

    result = solve_problem(dataset)

    if result is None:
        return jsonify({"error": "No solution"})

    F, indices, x_vec = result

    return jsonify({
        "F": round(F, 4),
        "indices": indices,
        "x": x_vec
    })


if __name__ == "__main__":
    multiprocessing.freeze_support()
    webbrowser.open("http://127.0.0.1:8000")
    app.run(host="127.0.0.1", port=8000)