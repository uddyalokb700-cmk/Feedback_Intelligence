/**
 * Customer Feedback Intelligence (CFI)
 * Centralized API client with CSRF token injection and error formatting.
 */

function getCookie(name) {
  let cookieValue = null;
  if (document.cookie && document.cookie !== "") {
    const cookies = document.cookie.split(";");
    for (let i = 0; i < cookies.length; i++) {
      const cookie = cookies[i].trim();
      if (cookie.substring(0, name.length + 1) === name + "=") {
        cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
        break;
      }
    }
  }
  return cookieValue;
}

async function apiFetch(url, options = {}) {
  const headers = options.headers || {};
  const csrfToken = getCookie("csrftoken");

  if (csrfToken && !["GET", "HEAD"].includes((options.method || "GET").toUpperCase())) {
    headers["X-CSRFToken"] = csrfToken;
  }

  // If body is plain object and not FormData, serialize to JSON
  if (options.body && !(options.body instanceof FormData) && typeof options.body === "object") {
    headers["Content-Type"] = "application/json";
    options.body = JSON.stringify(options.body);
  }

  options.headers = headers;

  try {
    const response = await fetch(url, options);

    if (response.status === 401 || response.status === 403) {
      if (response.redirected) {
        window.location.href = response.url;
        return;
      }
    }

    const contentType = response.headers.get("content-type");
    let data = null;
    if (contentType && contentType.includes("application/json")) {
      data = await response.json();
    } else if (contentType && contentType.includes("text/csv")) {
      data = await response.blob();
    } else {
      data = await response.text();
    }

    if (!response.ok) {
      let errorMessage = "An unexpected error occurred.";
      if (data && typeof data === "object") {
        if (data.detail) errorMessage = data.detail;
        else if (data.error) errorMessage = data.error;
        else if (data.message) errorMessage = data.message;
        else errorMessage = Object.entries(data).map(([k, v]) => `${k}: ${Array.isArray(v) ? v.join(", ") : v}`).join("\n");
      }
      const err = new Error(errorMessage);
      err.status = response.status;
      err.data = data;
      throw err;
    }

    return data;
  } catch (error) {
    console.error("API error for", url, error);
    throw error;
  }
}
