import os
import tempfile
import threading
import logging
from math import ceil
from typing import Tuple
from flask import Flask, request, jsonify, make_response, abort
import pandas as pd

# ===========================================
# CONFIGURATION
# ===========================================
CSV_PATH = os.environ.get("FRIENDS_CSV", "friends_data.csv")
RAW_CSV_URL = os.environ.get(
    "RAW_CSV_URL",
    "https://github.com/gchandra10/filestorage/raw/main/friends_data.csv"
)
LOG_FILE = "app.log"

csv_lock = threading.Lock()

# ===========================================
# LOGGING SETUP
# ===========================================
logger = logging.getLogger("friends_api")
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

fh = logging.FileHandler(LOG_FILE)
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
logger.addHandler(fh)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger.addHandler(ch)

# ===========================================
# FLASK APP
# ===========================================
app = Flask(__name__)

# ===========================================
# HELPER FUNCTIONS
# ===========================================
def ensure_csv_exists(path: str) -> None:
    """Ensure CSV exists; download or create empty if missing."""
    if os.path.exists(path):
        return
    logger.info(f"{path} not found locally. Attempting to download from {RAW_CSV_URL}")
    try:
        import urllib.request
        urllib.request.urlretrieve(RAW_CSV_URL, path)
        logger.info("Downloaded CSV from repository raw URL.")
    except Exception as e:
        logger.warning(f"Failed to download CSV: {e}. Creating empty CSV placeholder.")
        df = pd.DataFrame(columns=["id", "first_name", "last_name"])
        df.to_csv(path, index=False)


def read_csv_safe(path: str) -> pd.DataFrame:
    """Safely read CSV with lock and ensure id column exists."""
    ensure_csv_exists(path)
    with csv_lock:
        try:
            df = pd.read_csv(path, dtype=str)
        except pd.errors.EmptyDataError:
            df = pd.DataFrame(columns=["id", "first_name", "last_name"])

    if "id" not in df.columns:
        df.insert(0, "id", [str(i + 1) for i in range(len(df))])
        write_csv_safe(df, path)

    return df


def write_csv_safe(df: pd.DataFrame, path: str) -> None:
    """Safely write CSV atomically."""
    with csv_lock:
        dirpath = os.path.dirname(os.path.abspath(path)) or "."
        fd, tmp_path = tempfile.mkstemp(dir=dirpath, prefix="tmp_", suffix=".csv")
        os.close(fd)
        try:
            df.to_csv(tmp_path, index=False)
            os.replace(tmp_path, path)
            logger.debug(f"Wrote CSV atomically to {path}")
        except Exception as e:
            logger.exception("Failed to write CSV atomically")
            try:
                os.remove(tmp_path)
            except Exception:
                pass
            raise


def find_record_by_id(df: pd.DataFrame, rec_id: str) -> Tuple[int, pd.Series]:
    """Return (index, row) for record with given id."""
    matches = df.index[df["id"].astype(str) == str(rec_id)].tolist()
    if not matches:
        raise KeyError(f"No record with id={rec_id}")
    idx = matches[0]
    return idx, df.loc[idx]


def paginate_df(df: pd.DataFrame, page: int, per_page: int) -> Tuple[pd.DataFrame, dict]:
    """Paginate DataFrame."""
    total = len(df)
    per_page = max(1, min(per_page, 100))
    page = max(1, page)
    total_pages = max(1, ceil(total / per_page))
    start = (page - 1) * per_page
    end = start + per_page
    page_df = df.iloc[start:end]
    meta = {
        "total": total,
        "page": page,
        "per_page": per_page,
        "total_pages": total_pages,
        "count": len(page_df),
    }
    return page_df, meta


# ===========================================
# ERROR HANDLERS
# ===========================================
@app.errorhandler(400)
def bad_request(e):
    logger.warning(f"400: {e}")
    return jsonify({"error": "Bad Request", "message": str(e)}), 400


@app.errorhandler(404)
def not_found(e):
    logger.info(f"404: {e}")
    return jsonify({"error": "Not Found", "message": str(e)}), 404


@app.errorhandler(500)
def server_error(e):
    logger.exception(f"500: {e}")
    return (
        jsonify({"error": "Internal Server Error", "message": "An unexpected error occurred."}),
        500,
    )


# ===========================================
# ROUTES
# ===========================================
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/characters", methods=["GET"])
def list_characters():
    """GET /characters?page=1&per_page=10"""
    try:
        page = int(request.args.get("page", 1))
        per_page = int(request.args.get("per_page", 10))
    except ValueError:
        abort(make_response(jsonify({"error": "page and per_page must be integers"}), 400))

    df = read_csv_safe(CSV_PATH)
    page_df, meta = paginate_df(df, page, per_page)
    data = page_df.fillna("").to_dict(orient="records")
    return jsonify({"meta": meta, "data": data}), 200


@app.route("/characters/search", methods=["GET"])
def search_characters():
    """GET /characters/search?first_name=Phoebe or ?last_name=Geller"""
    first = request.args.get("first_name")
    last = request.args.get("last_name")
    if not first and not last:
        return jsonify({"error": "Provide first_name or last_name query parameter."}), 400

    df = read_csv_safe(CSV_PATH)
    mask = pd.Series([False] * len(df))
    if first:
        mask |= df["first_name"].astype(str).str.contains(first, case=False, na=False)
    if last:
        mask |= df["last_name"].astype(str).str.contains(last, case=False, na=False)

    results = df[mask].fillna("").to_dict(orient="records")
    return jsonify({"meta": {"query_count": len(results)}, "data": results}), 200


@app.route("/characters/<int:rec_id>", methods=["PUT"])
def update_character(rec_id):
    """PUT /characters/<id>"""
    if not request.is_json:
        return jsonify({"error": "Request body must be JSON"}), 400

    payload = request.get_json()
    if not payload:
        return jsonify({"error": "JSON body with at least one field is required"}), 400

    df = read_csv_safe(CSV_PATH)
    try:
        idx, _ = find_record_by_id(df, str(rec_id))
    except KeyError:
        return jsonify({"error": f"Character with id {rec_id} not found"}), 404

    for k, v in payload.items():
        if k == "id":
            continue
        if k not in df.columns:
            df[k] = ""
        df.at[idx, k] = str(v) if v is not None else ""

    try:
        write_csv_safe(df, CSV_PATH)
        return jsonify({"message": "Updated", "data": df.loc[idx].to_dict()}), 200
    except Exception:
        logger.exception("Failed to persist updated character")
        return jsonify({"error": "Failed to persist update"}), 500


@app.route("/characters/<int:rec_id>", methods=["DELETE"])
def delete_character(rec_id):
    """DELETE /characters/<id>"""
    try:
        df = read_csv_safe(CSV_PATH)
        if "id" not in df.columns:
            return jsonify({"error": "CSV missing id column"}), 500

        matches = df.index[df["id"].astype(str) == str(rec_id)].tolist()
        if not matches:
            return jsonify({"error": f"Character with id {rec_id} not found"}), 404

        # Drop and renumber safely
        df = df.drop(index=matches[0]).reset_index(drop=True)
        df["id"] = [str(i + 1) for i in range(len(df))]

        write_csv_safe(df, CSV_PATH)
        return "", 204

    except Exception as e:
        logger.exception("Failed to persist deletion")
        return jsonify({"error": "Failed to persist deletion", "details": str(e)}), 500


# ===========================================
# RUN APP
# ===========================================
if __name__ == "__main__":
    logger.info("Starting Flask server on http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
