const API_URL = "http://localhost:8000";
const chatLog = document.getElementById('chat-log');
const form = document.getElementById('controls');
const input = document.getElementById('message-input');
const imageInput = document.getElementById('image-upload');
const voiceBtn = document.getElementById('voice-btn');

const toggleBtn = document.getElementById('theme-toggle');
const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
if (prefersDark) document.body.classList.add('dark');

toggleBtn.onclick = () => {
  document.body.classList.toggle('dark');
  localStorage.setItem('chat-theme',
    document.body.classList.contains('dark') ? 'dark' : 'light');
};
const saved = localStorage.getItem('chat-theme');
if (saved) document.body.classList.toggle('dark', saved === 'dark');

document.getElementById('chat-container').style.display = "";
const chatContainer = document.getElementById('chat-container');
chatContainer.style.display = "";
void chatContainer.offsetHeight; // Force reflow

let user_id = localStorage.getItem('user_id'); // demo only
if (!user_id) {
  user_id = prompt("Please enter your user ID:");
  if (!user_id) {
    user_id = "user" + Math.floor(Math.random() * 10000);
  }
  localStorage.setItem('user_id', user_id);
}

window.onload = async () => {
  await loadThreads();
  const threadList = document.querySelectorAll('#thread-list li');
  if (threadList.length > 0) {
    // Auto-select the first thread
    threadList[0].click();
  } else {
    // No threads: create a new one and select it
    createAndSelectNewThread();
  }
};
async function createAndSelectNewThread() {
  thread_id = "thread-" + Date.now();
  // Optimistically add to UI
  await loadThreads();
  selectThread(thread_id);
}


let thread_id = null; // currently selected thread
document.getElementById('login-btn').onclick = function() {
  const input = document.getElementById('user-id-input');
  const uid = input.value.trim();
  if (uid) {
    user_id = uid;
    localStorage.setItem('user_id', user_id);
    document.getElementById('main-container').style.display = "";
    document.getElementById('login-container').style.display = "none";
    // Optionally: loadThreads();
  }
};

async function loadThreads() {
  const res = await fetch(`${API_URL}/threads?user_id=${user_id}`);
  const threads = await res.json();
  const threadList = document.getElementById('thread-list');
  threadList.innerHTML = '';
  threads.forEach(tid => {
    const li = document.createElement('li');
    li.textContent = tid;
    li.onclick = () => selectThread(tid);
    if (tid === thread_id) li.classList.add('selected');
    threadList.appendChild(li);
  });
}
async function selectThread(tid) {
  thread_id = tid;
  await loadThreads();
  // Load messages for this thread
  const res = await fetch(`${API_URL}/threads/${tid}/messages?user_id=${user_id}&thread_id=${tid}`);
  const messages = await res.json();
  chatLog.innerHTML = '';
  messages.forEach(msg => addMessage(msg.role === "assistant" ? "bot" : "user", msg.content));
}

function addMessage(role, text) {
  const div = document.createElement('div');
  div.className = `message ${role}`;
  if (role === "bot") {
    // Render markdown to HTML for bot messages
    div.innerHTML = marked.parse(text);
  } else {
    div.textContent = `${role === "user" ? "🧑" : "🤖"} ${text}`;
  }
  chatLog.appendChild(div);
  chatLog.scrollTop = chatLog.scrollHeight;
}




async function sendImageMessage(message, file) {
  addMessage("user", `${message} [Image: ${file.name}]`);
  input.value = "";
  addMessage("bot", "...");
  const formData = new FormData();
  formData.append("message", message);
  formData.append("image", file);
  const res = await fetch(`${API_URL}/chat-image`, {
    method: "POST",
    body: formData
  });
  const data = await res.json();
  chatLog.lastChild.textContent = data.response;
}

async function sendVoice(file) {
  addMessage("user", "[Voice message]");
  addMessage("bot", "...");
  const formData = new FormData();
  formData.append("audio", file);
  const res = await fetch(`${API_URL}/chat-voice`, {
    method: "POST",
    body: formData
  });
  const data = await res.json();
  chatLog.lastChild.textContent = data.response;
}

// Form submit: text OR image
form.onsubmit = async e => {
  e.preventDefault();
  if (!thread_id) {
    alert("Select or start a chat thread first.");
    return;
  }

  // Optionally, disable input to prevent double-submit
  input.disabled = true;

  const formData = new FormData();
  formData.append("message", input.value || "(no text)");
  formData.append("user_id", user_id);
  formData.append("thread_id", thread_id);
  if (imageInput.files[0]) {
    formData.append("image", imageInput.files[0]);
  }

  addMessage("user", input.value + (imageInput.files[0] ? ` [Image: ${imageInput.files[0].name}]` : ''));
  input.value = "";
  addMessage("bot", "...");

  // Stream the response
  const res = await fetch(`${API_URL}/chat-multimodal`, {
    method: "POST",
    body: formData
  });

  if (!res.body) {
    chatLog.lastChild.textContent = "Error: No response stream.";
    input.disabled = false;
    return;
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let answerText = ""; 
  let raw = "";        // everything we’ve received
  let partial = "";
while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    // Look for the images marker in the streamed content
    const imgStart = buffer.indexOf('[[[IMAGES_JSON]]]');
    if (imgStart !== -1) {
        // All LLM text ends before the marker
        answerText = buffer.substring(0, imgStart);
        chatLog.lastChild.innerHTML = marked.parse(answerText);

        // Now try to extract the JSON
        const imgEnd = buffer.indexOf('[[[/IMAGES_JSON]]]', imgStart);
        if (imgEnd !== -1) {
            const imgJsonStr = buffer.substring(
                imgStart + '[[[IMAGES_JSON]]]'.length,
                imgEnd
            );
            let images = [];
            try {
                images = JSON.parse(imgJsonStr);
            } catch (e) {
                images = [];
            }
            // Render images below the chat message
            images.forEach(img => {
                const imgElem = document.createElement('img');
                imgElem.src = img.url || `/assets/images/${img.filename}`;
                imgElem.alt = img.caption || img.filename;
                imgElem.style.maxWidth = "200px";
                imgElem.style.display = "block";
                chatLog.appendChild(imgElem);

                // Optionally add caption
                if (img.caption) {
                    const captionElem = document.createElement('div');
                    captionElem.textContent = img.caption;
                    chatLog.appendChild(captionElem);
                }
            });
        }
        break;
    } else {
        // Normal streaming (no marker yet)
        partial += decoder.decode(value, { stream: true });
        const lines = partial.split('\n');
        partial = lines.pop(); 
        raw += lines.join('\n') + '\n'; 
        chatLog.lastChild.innerHTML = marked.parse(raw);
        chatLog.scrollTop = chatLog.scrollHeight;
    }
}
// In case the stream ends without images marker, still finalize with markdown
if (!answerText && buffer) {
    chatLog.lastChild.innerHTML = marked.parse(buffer);
}
if (partial) {
  raw += partial;
  chatLog.lastChild.innerHTML = marked.parse(raw);
}



  // Clean up and prepare for next input
  input.disabled = false;
  imageInput.value = "";
  input.focus();
};


// Voice input
let recognition;
voiceBtn.onclick = () => {
  if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
    alert("Voice not supported in this browser.");
    return;
  }
  if (!recognition) {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    recognition = new SpeechRecognition();
    recognition.lang = 'en-US';
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;
    recognition.onresult = (event) => {
      input.value = event.results[0][0].transcript;
      form.dispatchEvent(new Event('submit'));
    };
    recognition.onerror = (event) => {
      alert("Voice error: " + event.error);
    };
  }
  recognition.start();
};

// Optional: Handle paste images
input.addEventListener('paste', (e) => {
  const items = e.clipboardData.items;
  for (let item of items) {
    if (item.type.indexOf('image') === 0) {
      const file = item.getAsFile();
      sendImageMessage(input.value || "(pasted image)", file);
      e.preventDefault();
    }
  }
});
document.getElementById('new-thread-btn').onclick = async () => {
  const res = await fetch(`${API_URL}/threads`, {
    method: "POST",
    body: new URLSearchParams({ user_id }),
  });
  const data = await res.json();
  thread_id = data.thread_id;
  await loadThreads();
  selectThread(thread_id);
};
