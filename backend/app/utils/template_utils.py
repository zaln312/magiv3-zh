import json
import urllib.request
import urllib.error


def deep_get(obj: dict, path: str, default=None):
    keys = path.split(".")
    current = obj
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        elif isinstance(current, list) and key.isdigit():
            idx = int(key)
            if idx < len(current):
                current = current[idx]
            else:
                return default
        else:
            return default
    return current


def replace_placeholders(template, prose, images_base64):
    if isinstance(template, str):
        return template.replace("{prose}", prose)
    if isinstance(template, dict):
        result = {}
        for k, v in template.items():
            result[k] = replace_placeholders(v, prose, images_base64)
        return result
    if isinstance(template, list):
        return [replace_placeholders(item, prose, images_base64) for item in template]
    return template


def check_images_marker(val, images_base64):
    if isinstance(val, str) and val == "{images_base64}":
        return images_base64
    if isinstance(val, dict):
        result = {}
        for k, v in val.items():
            result[k] = check_images_marker(v, images_base64)
        return result
    if isinstance(val, list):
        return [check_images_marker(item, images_base64) for item in val]
    return val


def execute_user_code(code: str, func_name: str, **kwargs):
    namespace = {}
    exec(code, namespace)
    if func_name not in namespace:
        raise RuntimeError(f"用户代码中未定义 {func_name} 函数")
    return namespace[func_name](**kwargs)


def http_request(method: str, url: str, headers: dict, body: dict | None = None):
    data_bytes = None
    if body is not None:
        data_bytes = json.dumps(body, ensure_ascii=False).encode("utf-8")
        if "Content-Type" not in headers:
            headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url, data=data_bytes, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            resp_data = resp.read().decode("utf-8")
            return json.loads(resp_data)
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace") if e.fp else ""
        raise RuntimeError(f"HTTP {e.code}: {err_body[:500]}")
    except Exception as e:
        raise RuntimeError(f"请求失败: {str(e)}")