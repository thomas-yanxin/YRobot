// Reachy Mini Live Chat — web UI client.
const $ = (id) => document.getElementById(id);
const log = $("log"), stateEl = $("state"), latEl = $("latency"), emoEl = $("emotion");

let lastBot = null; // coalesce streamed assistant clauses into one bubble

function bubble(cls, text, emotion) {
  const div = document.createElement("div");
  div.className = "bubble " + cls;
  if (emotion) {
    const chip = document.createElement("span");
    chip.className = "emochip";
    chip.textContent = "〔" + emotion + "〕";
    div.appendChild(chip);
  }
  div.appendChild(document.createTextNode(text));
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
  return div;
}

function setState(s) {
  stateEl.textContent = s;
  stateEl.className = "pill state-" + s;
  if (s !== "speaking") lastBot = null; // next assistant text starts a fresh bubble
}

function handle(evt) {
  switch (evt.kind) {
    case "state": setState(evt.state); break;
    case "user": lastBot = null; bubble("user", evt.text); break;
    case "assistant":
      if (lastBot) { lastBot.appendChild(document.createTextNode(evt.text)); }
      else { lastBot = bubble("bot", evt.text); }
      log.scrollTop = log.scrollHeight;
      break;
    case "emotion":
      emoEl.textContent = "😊 " + evt.name; emoEl.classList.remove("hidden");
      clearTimeout(emoEl._t); emoEl._t = setTimeout(() => emoEl.classList.add("hidden"), 2500);
      break;
    case "latency":
      if (evt.e2e_ms != null) latEl.textContent = Math.round(evt.e2e_ms) + " ms";
      break;
    case "system": bubble("sys", evt.text); break;
  }
}

// --- SSE event stream ---
function connect() {
  const es = new EventSource("/api/events");
  es.onmessage = (e) => { try { handle(JSON.parse(e.data)); } catch (_) {} };
  es.onerror = () => { es.close(); setTimeout(connect, 1500); };
}
connect();

// --- text input ---
$("composer").addEventListener("submit", async (e) => {
  e.preventDefault();
  const input = $("msg"), text = input.value.trim();
  if (!text) return;
  input.value = "";
  await fetch("/api/say", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
});

// --- camera preview (poll ~8fps) ---
const cam = $("cam"), camoff = $("camoff");
setInterval(() => {
  const url = "/api/frame.jpg?t=" + Date.now();
  const probe = new Image();
  probe.onload = () => { cam.src = url; camoff.style.display = "none"; };
  probe.onerror = () => { camoff.style.display = "grid"; };
  probe.src = url;
}, 125);
