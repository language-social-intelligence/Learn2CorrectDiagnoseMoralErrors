"use strict";
const $ = id => document.getElementById(id);
const pause = ms => new Promise(resolve => setTimeout(resolve, ms));
const chats = [];
let info = null, connected = false, activeChat = null, refreshing = false;
let preferredMode = $("review-mode").value;
function chatBusy() { return activeChat?.turns.some(turn => turn.pending) || false; }
const stageLabels = {
  queued: "Waiting for an available slot…",
  generating: "Thinking",
  diagnosing: "Aligning",
  paraphrasing: "Responding"
};
function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}
function announce(text) { $("announcement").textContent = text; }
function scrollToBottom() { $("chat-scroll").scrollTop = $("chat-scroll").scrollHeight; }
function nearBottom() {
  const box = $("chat-scroll");
  return box.scrollHeight - box.scrollTop - box.clientHeight < 180;
}
function updateControls() {
  const busy = chatBusy();
  $("submit").disabled = busy || !connected || info?.status !== "ready" || !$("question").value.trim();
  $("submit").setAttribute("aria-label", busy ? "Response in progress" : "Send message");
  $("review-mode").disabled = busy;
  document.querySelectorAll(".examples button, .edit-question").forEach(button => button.disabled = busy);
}
function resizeInput() {
  const input = $("question");
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 180) + "px";
  $("count").textContent = input.value.length.toLocaleString() + " / 4,000";
  $("count").hidden = input.value.length < 3000;
  updateControls();
}
function fillQuestion(question) {
  if (chatBusy()) return;
  $("question").value = question;
  resizeInput();
  $("question").focus();
}
function toggleNavigation(open) {
  const mobile = matchMedia("(max-width: 720px)").matches;
  document.body.classList.toggle("sidebar-open", mobile && open);
  $("nav-backdrop").hidden = !mobile || !open;
  $("toggle-sidebar").setAttribute("aria-expanded", String(mobile && open));
  $("toggle-sidebar").setAttribute("aria-label", open && mobile ? "Close navigation" : "Open navigation");
  $("sidebar").inert = mobile && !open;
}
function renderChatList() {
  $("chat-list").replaceChildren();
  const populated = chats.filter(chat => chat.turns.length).slice().reverse();
  $("no-chats").hidden = populated.length > 0;
  for (const chat of populated) {
    const button = element("button", "chat-link", chat.title);
    button.type = "button";
    button.title = chat.title;
    button.setAttribute("aria-current", String(chat === activeChat));
    button.addEventListener("click", () => {
      activeChat.draft = $("question").value;
      activeChat = chat;
      renderConversation();
      toggleNavigation(false);
    });
    $("chat-list").append(button);
  }
  updateControls();
}
function createChat() {
  if (activeChat) activeChat.draft = $("question").value;
  if (!activeChat || activeChat.turns.length) {
    activeChat = {title: "New chat", turns: [], draft: "", reviewMode: preferredMode};
    chats.push(activeChat);
  } else {
    activeChat.draft = "";
  }
  renderConversation();
  toggleNavigation(false);
  $("question").focus();
}
// Use text nodes for model output; only bold and inline code receive formatting.
function appendAnswer(node, text) {
  const pattern = /\*\*([^*\n]+)\*\*|`([^`\n]+)`/g;
  let cursor = 0;
  for (const match of text.matchAll(pattern)) {
    node.append(document.createTextNode(text.slice(cursor, match.index)));
    node.append(element(match[1] !== undefined ? "strong" : "code", "", match[1] ?? match[2]));
    cursor = match.index + match[0].length;
  }
  node.append(document.createTextNode(text.slice(cursor)));
}
function renderAssistant(turn) {
  const box = turn.assistant;
  box.replaceChildren();
  box.setAttribute("aria-busy", String(!turn.result && !turn.error));
  const header = element("div", "assistant-header");
  const avatar = element("span", "assistant-avatar", "h");
  avatar.setAttribute("aria-hidden", "true");
  header.append(avatar, element("span", "", "Human-scale AI"));
  box.append(header);
  if (turn.result) {
    const result = turn.result;
    if (result.truncated) box.append(element("p", "notice", "The response reached the generation limit and may be incomplete."));
    const answer = element("div", "answer");
    appendAnswer(answer, result.final_answer);
    box.append(answer);
    const actions = element("div", "response-actions");
    const copy = element("button", "copy-answer", "Copy");
    copy.type = "button";
    copy.setAttribute("aria-label", "Copy response");
    copy.addEventListener("click", async () => {
      try { await navigator.clipboard.writeText(result.final_answer); copy.textContent = "Copied"; }
      catch { copy.textContent = "Select text to copy"; }
    });
    actions.append(copy);
    box.append(actions);
  } else if (turn.error) {
    box.append(element("p", "request-error", turn.error));
    const edit = element("button", "edit-question", "Edit question");
    edit.type = "button";
    edit.addEventListener("click", () => fillQuestion(turn.question));
    box.append(edit);
  } else {
    const index = {queued: -1, generating: 0, diagnosing: 1, paraphrasing: 2}[turn.stage] ?? -1;
    const steps = element("ol", "steps");
    steps.setAttribute("aria-label", "Response progress");
    ["1. Thinking", "2. Aligning", "3. Responding"].forEach((text, i) => {
      const step = element("li", i < index ? "done" : i === index ? "active" : "", text);
      if (i === index) step.setAttribute("aria-current", "step");
      steps.append(step);
    });
    const pending = element("div", "pending");
    pending.append(steps);
    if (index < 0) pending.append(element("p", "status-text", stageLabels.queued));
    box.append(pending);
  }
}
function appendTurn(turn) {
  const article = element("article", "turn");
  const user = element("div", "user-message");
  user.setAttribute("aria-label", "Your message");
  user.append(element("div", "user-bubble", turn.question));
  turn.assistant = element("div", "assistant-message");
  article.append(user, turn.assistant);
  renderAssistant(turn);
  $("conversation").append(article);
}
function renderConversation() {
  $("conversation").replaceChildren();
  $("welcome").hidden = activeChat.turns.length > 0;
  $("conversation").hidden = activeChat.turns.length === 0;
  for (const turn of activeChat.turns) appendTurn(turn);
  $("question").value = activeChat.draft;
  $("review-mode").value = activeChat.reviewMode;
  resizeInput();
  renderChatList();
  scrollToBottom();
}
async function api(path, options) {
  const response = await fetch(path, {...options, signal: AbortSignal.timeout(15000)});
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "The request could not be completed.");
  return data;
}
async function refresh() {
  if (refreshing) return;
  refreshing = true;
  try {
    const data = await api("/api/info");
    if (!info) {
      for (const example of data.examples || []) {
        const button = element("button", "", example.label);
        button.type = "button";
        button.addEventListener("click", () => fillQuestion(example.values.question));
        $("examples").append(button);
      }
    }
    info = data;
    connected = true;
    $("connection").textContent = data.status === "ready" ? "Online" : data.status === "loading" ? "Starting…" : "Unavailable";
  } catch {
    connected = false;
    $("connection").textContent = "Connection unavailable";
  }
  $("connection").classList.toggle("online", connected && info?.status === "ready");
  refreshing = false;
  updateControls();
}
$("form").addEventListener("submit", async event => {
  event.preventDefault();
  const question = $("question").value.trim();
  if (chatBusy() || !connected || info?.status !== "ready" || !question) return;
  const chat = activeChat;
  const reviewMode = chat.reviewMode;
  const turn = {question, reviewMode, pending: true, stage: "queued", createdAt: new Date().toISOString()};
  chat.turns.push(turn);
  if (chat.turns.length === 1) chat.title = question;
  chat.draft = "";
  renderConversation();
  announce("Preparing your response.");
  try {
    const job = await api("/api/generate", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({question, review_mode: reviewMode})});
    const deadline = Date.now() + 15 * 60 * 1000;
    while (Date.now() < deadline) {
      const state = await api("/api/results/" + encodeURIComponent(job.id));
      if (state.status === "error") throw new Error(state.error);
      if (state.status === "complete") {
        turn.result = state.result;
        if (activeChat === chat) {
          const follow = nearBottom();
          renderAssistant(turn);
          if (follow) scrollToBottom();
          announce("Response complete.");
        }
        break;
      }
      const stage = state.stage || state.status;
      if (stage !== turn.stage) {
        turn.stage = stage;
        if (activeChat === chat) {
          renderAssistant(turn);
          announce(stageLabels[stage] || stageLabels.queued);
        }
      }
      await pause(500);
    }
    if (!turn.result) throw new Error("The request took too long. Please try again later.");
  } catch (error) {
    turn.error = error.name === "TimeoutError" ? "The connection timed out. Please try again." : error.message;
    if (activeChat === chat) {
      renderAssistant(turn);
      announce(turn.error);
    }
  } finally {
    turn.pending = false;
    updateControls();
    await refresh();
  }
});
$("question").addEventListener("input", () => {
  activeChat.draft = $("question").value;
  resizeInput();
});
$("question").addEventListener("keydown", event => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    if (!$("submit").disabled) $("form").requestSubmit();
  }
});
$("new-chat").addEventListener("click", createChat);
$("review-mode").addEventListener("change", () => {
  activeChat.reviewMode = $("review-mode").value;
  preferredMode = activeChat.reviewMode;
});
$("toggle-sidebar").addEventListener("click", () => toggleNavigation(!document.body.classList.contains("sidebar-open")));
$("nav-backdrop").addEventListener("click", () => toggleNavigation(false));
matchMedia("(max-width: 720px)").addEventListener("change", () => toggleNavigation(false));
document.addEventListener("keydown", event => {
  if (event.key === "Escape" && document.body.classList.contains("sidebar-open")) {
    toggleNavigation(false);
    $("toggle-sidebar").focus();
  }
});
document.querySelectorAll(".about-trigger").forEach(button => button.addEventListener("click", () => $("about").showModal()));
$("close-about").addEventListener("click", () => $("about").close());
$("about").addEventListener("click", event => {
  if (event.target === $("about")) {
    const bounds = $("about").getBoundingClientRect();
    if (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom) $("about").close();
  }
});
createChat();
refresh();
setInterval(refresh, 10000);
