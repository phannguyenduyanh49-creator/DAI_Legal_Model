export function safeSourceUrl(value) {
  if (typeof value !== "string" || !value.trim()) return "";
  try {
    const url = new URL(value);
    return ["https:", "http:"].includes(url.protocol) ? url.href : "";
  } catch {
    return "";
  }
}

export function answerOutcomeNotice(outcome, verifierStatus) {
  if (outcome === "answered_unverified") {
    return {
      tone: "warning",
      text: "Câu trả lời tham khảo — chưa được xác minh đầy đủ.",
    };
  }
  if (outcome === "needs_clarification") {
    return {
      tone: "info",
      text: "Cần thêm thông tin hoặc quyền tra cứu trước khi đưa ra câu trả lời.",
    };
  }
  if (outcome === "withheld") {
    return {
      tone: "warning",
      text: "Hệ thống đã giữ lại câu trả lời vì chưa đáp ứng điều kiện Evidence hoặc dữ liệu thiết yếu.",
    };
  }
  if (outcome === "verification_failed") {
    return {
      tone: "error",
      text: "Câu trả lời không vượt qua kiểm tra integrity hoặc verifier; nội dung đã được giữ lại.",
    };
  }
  if (outcome === "FAILED" || (verifierStatus && verifierStatus !== "passed")) {
    return {
      tone: "warning",
      text: "Yêu cầu đã được xử lý nhưng câu trả lời chưa được xác minh.",
    };
  }
  return null;
}
