import os
import re
import sys
from urllib.parse import urlparse

import yaml


REQUIRED_IMAGE_FILES = [
    "docs/reward_curve.png",
    "docs/loss_curve.png",
]

REQUIRED_EXISTENCE_FILES = [
    "openenv.yaml",
    "train_llm.py",
    "notebooks/train_colab.ipynb",
    "README.md",
]


def fail(msg):
    print(f"[FAIL] {msg}")
    return False


def ok(msg):
    print(f"[OK]   {msg}")
    return True


def is_valid_external_url(url):
    parsed = urlparse(url)
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)


def check_files_exist():
    status = True
    for path in REQUIRED_EXISTENCE_FILES + REQUIRED_IMAGE_FILES:
        if os.path.exists(path):
            ok(f"File exists: {path}")
        else:
            status = fail(f"Missing required file: {path}") and status
    return status


def check_openenv_yaml():
    status = True
    try:
        with open("openenv.yaml", "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except Exception as e:
        return fail(f"Could not parse openenv.yaml: {e}")

    required_keys = ["name", "description", "version", "entrypoint"]
    for key in required_keys:
        if key in data and data[key]:
            ok(f"openenv.yaml has key: {key}")
        else:
            status = fail(f"openenv.yaml missing key: {key}") and status

    entrypoint = data.get("entrypoint", "")
    if ":" not in entrypoint:
        status = fail("openenv.yaml entrypoint must be in file:Class format") and status
    else:
        file_name, class_name = entrypoint.split(":", 1)
        py_file = f"{file_name}.py"
        if os.path.exists(py_file):
            ok(f"Entrypoint file exists: {py_file}")
            with open(py_file, "r", encoding="utf-8") as f:
                contents = f.read()
            class_pattern = rf"class\s+{re.escape(class_name)}\s*\("
            if re.search(class_pattern, contents):
                ok(f"Entrypoint class exists: {class_name}")
            else:
                status = fail(f"Entrypoint class missing in {py_file}: {class_name}") and status
        else:
            status = fail(f"Entrypoint file missing: {py_file}") and status

    return status


def check_readme_links():
    status = True
    with open("README.md", "r", encoding="utf-8") as f:
        content = f.read()

    links = re.findall(r"\[([^\]]+)\]\(([^)]+)\)", content)
    if not links:
        return fail("README has no markdown links")

    hf_links = [url for text, url in links if "hugging face" in text.lower() or "space" in text.lower()]
    writeup_links = [url for text, url in links if "writeup" in text.lower() or "blog" in text.lower() or "video" in text.lower() or "slides" in text.lower()]

    if hf_links and any(is_valid_external_url(url) for url in hf_links):
        ok("README includes an external Hugging Face Space-style link")
    else:
        status = fail("README is missing a valid external HF Space link") and status

    if writeup_links and any(is_valid_external_url(url) for url in writeup_links):
        ok("README includes an external writeup/video/slides link")
    else:
        status = fail("README is missing a valid external writeup link") and status

    if "(notebooks/train_colab.ipynb)" in content or "notebooks/train_colab.ipynb" in content:
        ok("README links the training notebook")
    else:
        status = fail("README missing notebook link") and status

    return status


def main():
    print("Running OpenEnv hackathon pre-submit checks...")
    checks = [
        check_files_exist(),
        check_openenv_yaml(),
        check_readme_links(),
    ]
    if all(checks):
        print("\nAll checks passed.")
        sys.exit(0)
    print("\nOne or more checks failed. Fix before submission.")
    sys.exit(1)


if __name__ == "__main__":
    main()
