const $ = (sel) => document.querySelector(sel);

function switchTab(name) {
  document.querySelectorAll(".tab").forEach((t) => {
    const on = t.dataset.tab === name;
    t.classList.toggle("active", on);
    t.setAttribute("aria-selected", on ? "true" : "false");
  });
  document.querySelectorAll(".panel").forEach((p) =>
    p.classList.toggle("active", p.id === `panel-${name}`)
  );
}

document.querySelectorAll(".tab").forEach((btn) => {
  btn.addEventListener("click", () => switchTab(btn.dataset.tab));
});

function fillSelect(sel, values) {
  const el = $(sel);
  for (const v of values) {
    const opt = document.createElement("option");
    opt.value = v;
    opt.textContent = v;
    el.appendChild(opt);
  }
}

async function loadFacets() {
  const res = await fetch("/api/facets");
  const data = await res.json();
  fillSelect("#filter-material", data.materials || []);
  fillSelect("#filter-type", data.object_types || []);
  fillSelect("#filter-period", data.periods || []);
}

function openLightbox(item) {
  $("#lightbox-img").src = item.thumb_url;
  $("#lightbox-meta").textContent = JSON.stringify(
    {
      id: item.id,
      title: item.title,
      object_type: item.object_type,
      material: item.material,
      period: item.period,
      culture: item.culture,
      source: item.source,
      score: item.score,
      evidence: item.evidence,
      page_url: item.page_url,
    },
    null,
    2
  );
  $("#lightbox").classList.remove("hidden");
}

$("#lightbox-close").addEventListener("click", () => $("#lightbox").classList.add("hidden"));
$("#lightbox").addEventListener("click", (e) => {
  if (e.target.id === "lightbox") $("#lightbox").classList.add("hidden");
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") $("#lightbox").classList.add("hidden");
});

function escapeHtml(s) {
  return String(s || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function renderResults(payload) {
  const box = $("#results");
  box.innerHTML = "";
  const n = payload.results.length;
  $("#status").textContent =
    n === 0
      ? `No results · ${payload.method}`
      : `${n} ranked · ${payload.method} · ${payload.latency_ms} ms`;

  payload.results.forEach((item, i) => {
    const tile = document.createElement("button");
    tile.type = "button";
    tile.className = "tile";
    tile.style.animationDelay = `${Math.min(i, 18) * 0.035}s`;

    const ev = item.evidence || {};
    const chip = (key) => {
      const e = ev[key];
      if (!e || e.status === "unspecified") return "";
      const cls = e.status === "match" ? "ok" : "bad";
      const label = key === "object_type" ? "type" : key;
      return `<span class="chip ${cls}">${label} ${e.status}</span>`;
    };

    const title = escapeHtml(item.title || item.object_type || item.id);
    tile.innerHTML = `
      <div class="tile-frame">
        <span class="tile-rank">${String(i + 1).padStart(2, "0")}</span>
        <img src="${item.thumb_url}" alt="${title}" loading="lazy" />
      </div>
      <div class="tile-meta">
        <h3 class="tile-title">${title}</h3>
        <p class="tile-sub">${escapeHtml(item.period || "?")} · ${escapeHtml(item.material || "?")} · ${escapeHtml(item.object_type || "?")}</p>
        <p class="tile-score">${Number(item.score).toFixed(3)}</p>
        <div class="chips">${chip("period")}${chip("material")}${chip("object_type")}</div>
      </div>`;
    tile.addEventListener("click", () => openLightbox(item));
    box.appendChild(tile);
  });
}

$("#search-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const query = $("#query").value.trim();
  if (!query) return;
  $("#status").textContent = "Ranking gallery…";
  $("#results").innerHTML = "";
  const res = await fetch("/api/search", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query,
      method: $("#method").value,
      top_k: 24,
      material: $("#filter-material").value || null,
      object_type: $("#filter-type").value || null,
      period: $("#filter-period").value || null,
    }),
  });
  const data = await res.json();
  if (!res.ok) {
    $("#status").textContent = data.error || "Search failed";
    return;
  }
  renderResults(data);
});

async function loadMetrics() {
  const box = $("#metrics-table");
  try {
    const res = await fetch("/api/metrics");
    if (!res.ok) {
      box.textContent = "No evaluation yet. Run scripts/evaluate.py";
      return;
    }
    const data = await res.json();
    const sel = data.selected || {};
    const rows = (data.metrics || []).map(
      (m) =>
        `<tr><td>${m.method}</td><td>${m["P@5"].toFixed(4)}</td><td>${m["P@10"].toFixed(4)}</td><td>${m["R@10"].toFixed(4)}</td><td>${m.mAP.toFixed(4)}</td><td>${(m["nDCG@10"] ?? 0).toFixed(4)}</td></tr>`
    );
    box.innerHTML = `
      <p>Flagship <strong>compositional</strong> · α=${sel.alpha ?? "?"} · β=${sel.beta ?? "?"} · gallery ${data.n_gallery} · test ${data.n_test}</p>
      <table>
        <thead><tr><th>Method</th><th>P@5</th><th>P@10</th><th>R@10</th><th>mAP</th><th>nDCG@10</th></tr></thead>
        <tbody>${rows.join("")}</tbody>
      </table>`;
  } catch (err) {
    box.textContent = String(err);
  }
}

async function loadQueries() {
  const res = await fetch("/api/queries");
  const data = await res.json();
  const test = (data.queries || []).filter((q) => q.split === "test");
  const pick = $("#query-pick");
  pick.innerHTML = "";
  for (const q of test) {
    const opt = document.createElement("option");
    opt.value = q.id;
    opt.textContent = `${q.id}: ${q.text}`;
    pick.appendChild(opt);
  }
  const byId = Object.fromEntries(test.map((q) => [q.id, q]));
  function show() {
    const q = byId[pick.value];
    if (!q) return;
    $("#query-detail").innerHTML = `
      <p><strong>${escapeHtml(q.text)}</strong></p>
      <p>Constraints: <code>${escapeHtml(JSON.stringify(q.constraints))}</code></p>
      <p>Relevant images: ${q.relevant_ids.length}</p>
      <button type="button" id="run-query">Run this query</button>`;
    $("#run-query").onclick = () => {
      switchTab("search");
      $("#query").value = q.text;
      $("#search-form").requestSubmit();
    };
  }
  pick.addEventListener("change", show);
  show();
}

loadFacets().catch(console.error);
loadMetrics().catch(console.error);
loadQueries().catch(console.error);
