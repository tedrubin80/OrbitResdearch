# Session 2: Site & UX — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add scroll animations, storm-conditioned toggle table, prediction overlay on the globe, and mobile polish to orbitalchaos.online.

**Architecture:** All changes are in a single file (`public/index.html`). No build tools, no frameworks — pure CSS + vanilla JS using IntersectionObserver for animations, data attributes for toggle state, and Globe.GL's `arcsData` for prediction overlays.

**Tech Stack:** HTML, CSS, vanilla JavaScript, IntersectionObserver API, Globe.GL (already loaded via CDN).

**Prerequisite:** Session 1 results JSONs (`results/storm_conditioned_mae.json`, `results/sgp4_baselines.json`) should exist. If not yet available, use placeholder values marked with `TBD` and backfill later.

---

## File Structure

### Modified Files
| File | Changes |
|------|---------|
| `public/index.html:7-471` | CSS: animation classes, toggle styles, mobile fixes (~50 lines) |
| `public/index.html:487-504` | HTML: `data-count` attributes on stat values for counters |
| `public/index.html:505-575` | HTML: `animate-in` class on cards and headings |
| `public/index.html:579-654` | HTML: toggle-enabled results table with SGP4 row |
| `public/index.html:654-800` | JS: IntersectionObserver, counter, toggle logic, prediction overlay |

### New Files
| File | Responsibility |
|------|---------------|
| `public/predictions.json` | Mock prediction data for dev (overwritten by Session 3 cron) |

---

## Chunk 1: Scroll Animations

### Task 1: Fade-In and Stagger Animations

**Files:**
- Modify: `public/index.html` — CSS + HTML classes + JS

- [ ] **Step 1: Add animation CSS**

Add before `@media (max-width: 600px)` in `<style>`:

```css
.animate-in {
    opacity: 0;
    transform: translateY(20px);
    transition: opacity 0.6s ease-out, transform 0.6s ease-out;
}
.animate-in.visible {
    opacity: 1;
    transform: translateY(0);
}
```

- [ ] **Step 2: Add `animate-in` class to elements**

Add `animate-in` to: every `.stat-card`, `.spacecraft-card`, `.model-card`, `.weather-card`, `.pipeline-step`, `<h2>`, `.results-table`, `.chart-container`.

- [ ] **Step 3: Add IntersectionObserver JS**

Add at the start of the second `<script>` block (before ISS tracking code):

```javascript
(function() {
    var observer = new IntersectionObserver(function(entries) {
        entries.forEach(function(entry) {
            if (!entry.isIntersecting) return;
            var el = entry.target;
            var parent = el.parentElement;
            if (parent && (parent.classList.contains('stats-grid') ||
                parent.classList.contains('spacecraft-grid') ||
                parent.classList.contains('models-section') ||
                parent.classList.contains('weather-grid'))) {
                var siblings = Array.from(parent.querySelectorAll('.animate-in'));
                var idx = siblings.indexOf(el);
                if (idx > 0) el.style.transitionDelay = (idx * 0.12) + 's';
            }
            if (el.classList.contains('pipeline-step')) {
                var steps = Array.from(el.parentElement.querySelectorAll('.pipeline-step'));
                var stepIdx = steps.indexOf(el);
                if (stepIdx > 0) el.style.transitionDelay = (stepIdx * 0.15) + 's';
            }
            el.classList.add('visible');
            var counters = el.querySelectorAll('[data-count]');
            counters.forEach(animateCounter);
            observer.unobserve(el);
        });
    }, { threshold: 0.15 });
    document.querySelectorAll('.animate-in').forEach(function(el) { observer.observe(el); });
})();
```

- [ ] **Step 4: Verify — scroll through site, sections fade in, cards stagger**

- [ ] **Step 5: Commit**

```bash
git add public/index.html && git commit -m "feat: add scroll fade-in and stagger animations"
```

---

### Task 2: Counter Animation for Stats

- [ ] **Step 6: Add data attributes to stat values**

Replace stat card content with:
```html
<div class="stat-value" data-count="4800000" data-suffix="+" data-format="abbrev">0</div>
```
Similarly for `3`, `3 Years` (data-count="3" data-suffix=" Years"), `1 min` (data-count="1" data-suffix=" min").

- [ ] **Step 7: Add counter function**

```javascript
function animateCounter(el) {
    var target = parseInt(el.dataset.count);
    var suffix = el.dataset.suffix || '';
    var format = el.dataset.format || '';
    var duration = 1500;
    var start = performance.now();
    function update(now) {
        var elapsed = now - start;
        var progress = Math.min(elapsed / duration, 1);
        var eased = 1 - Math.pow(1 - progress, 3);
        var current = Math.round(eased * target);
        if (format === 'abbrev' && current >= 1000000) {
            el.textContent = (current / 1000000).toFixed(1) + 'M' + suffix;
        } else if (format === 'abbrev' && current >= 1000) {
            el.textContent = (current / 1000).toFixed(0) + 'K' + suffix;
        } else {
            el.textContent = current + suffix;
        }
        if (progress < 1) requestAnimationFrame(update);
    }
    requestAnimationFrame(update);
}
```

This is already called from the IntersectionObserver in Step 3 via `counters.forEach(animateCounter)`.

- [ ] **Step 8: Verify — stats count up from 0 when scrolled into view**

- [ ] **Step 9: Commit**

```bash
git add public/index.html && git commit -m "feat: add counter animation for stat values"
```

---

## Chunk 2: Storm-Conditioned Toggle Table

### Task 3: Interactive Toggle Results Table

**Files:**
- Modify: `public/index.html` — Results section HTML, toggle CSS, toggle JS

- [ ] **Step 10: Add toggle button CSS**

Add before `@media`:
```css
.toggle-btns { display: flex; gap: 8px; margin-bottom: 16px; flex-wrap: wrap; }
.toggle-btn {
    padding: 8px 16px; border-radius: 8px;
    border: 1px solid rgba(255,255,255,0.1);
    background: rgba(255,255,255,0.04); color: #90a4ae;
    font-size: 0.85em; cursor: pointer; transition: all 0.2s;
}
.toggle-btn:hover { border-color: rgba(79,195,247,0.3); color: #b0bec5; }
.toggle-btn.active { background: rgba(79,195,247,0.12); border-color: rgba(79,195,247,0.4); color: #4fc3f7; }
```

Add to `@media (max-width: 600px)`:
```css
.toggle-btns { gap: 6px; }
.toggle-btn { padding: 6px 12px; font-size: 0.8em; }
```

- [ ] **Step 11: Replace static Results HTML**

Replace the Results section (lines 579-654) with toggle buttons + dynamic table:

```html
<div class="section">
    <h2 class="animate-in">Results</h2>
    <p class="animate-in" style="color:#78909c;margin-bottom:16px;font-size:0.95em;">
        6-hour prediction MAE (Mean Absolute Error) in km. Trained on dual RTX 5090 GPUs via RunPod.
    </p>
    <div class="toggle-btns animate-in">
        <button class="toggle-btn active" onclick="toggleCondition('all',this)">All</button>
        <button class="toggle-btn" onclick="toggleCondition('quiet',this)">Quiet (Kp&le;3)</button>
        <button class="toggle-btn" onclick="toggleCondition('active',this)">Active (Kp 4-5)</button>
        <button class="toggle-btn" onclick="toggleCondition('storm',this)">Storm (Kp&ge;6)</button>
    </div>
    <table class="results-table animate-in" id="results-table">
        <thead><tr><th>Model</th><th>ISS (LEO)</th><th>DSCOVR (L1)</th><th>MMS-1 (HEO)</th></tr></thead>
        <tbody id="results-body"></tbody>
    </table>
    <div class="chart-container animate-in">
        <h3>MAE Comparison (log scale)</h3>
        <svg id="mae-chart" viewBox="0 0 700 260" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:700px;"></svg>
    </div>
</div>
```

- [ ] **Step 12: Add results data object and toggle JS**

Add the `RESULTS_DATA` object with `all` values from current table (126, 295, 175 for ISS etc.) and `TBD` placeholders for quiet/active/storm. Add `toggleCondition(condition, btn)` that:
1. Updates active button class
2. Builds table rows with DOM methods (createElement, textContent — no innerHTML with user data)
3. Highlights best values with `.best-value` class
4. Calls `updateChart(condition)` to rebuild the SVG

Use DOM manipulation methods (createElement, textContent, appendChild) for building table rows — avoid innerHTML with dynamic data.

For the SVG chart, since data is from a hardcoded trusted object (not user input), rebuilding with string concatenation is acceptable for the static SVG elements. Use the same log-scale bar width calculation as the current chart.

- [ ] **Step 13: Initialize table on page load**

Call `toggleCondition('all', null)` at the end of the JS to populate the table on load.

- [ ] **Step 14: Verify toggle works**

Click each button — table values and chart should update. "All" shows real numbers. Others show "TBD" until Session 1 data arrives.

- [ ] **Step 15: Commit**

```bash
git add public/index.html && git commit -m "feat: add storm-conditioned toggle table"
```

---

## Chunk 3: Prediction Overlay & Mobile Polish

### Task 4: Prediction Path on Globe

**Files:**
- Create: `public/predictions.json` (mock data)
- Modify: `public/index.html` — globe JS

- [ ] **Step 16: Create mock predictions.json**

JSON with `generated_at` (current ISO timestamp), `model: "lstm_iss_6h"`, and `path` array of ~30 objects with `lat, lng, alt, minutes_ahead` following an ISS-like ground track.

- [ ] **Step 17: Add prediction overlay function**

Add `loadPredictionOverlay()` that:
1. Fetches `predictions.json`
2. Checks freshness (`generated_at` < 2 hours ago)
3. Validates schema (lat -90 to 90, lng -180 to 180, alt 300-500)
4. Builds `arcsData` from consecutive point pairs
5. Renders dashed cyan arcs on the globe via `window.issGlobe.arcsData()`
6. Silently skips on any error

- [ ] **Step 18: Call from initGlobe()**

Add `loadPredictionOverlay()` at the end of `initGlobe()`.

- [ ] **Step 19: Verify — globe shows faint prediction arcs**

(Update `generated_at` in mock JSON to current time if needed for freshness check)

- [ ] **Step 20: Commit**

```bash
git add public/index.html public/predictions.json
git commit -m "feat: add prediction overlay to ISS globe"
```

---

### Task 5: Mobile Polish

- [ ] **Step 21: Add mobile CSS**

Inside `@media (max-width: 600px)`:
```css
.globe-container { touch-action: none; }
.weather-grid { grid-template-columns: repeat(2, 1fr); }
.toggle-btns { overflow-x: auto; -webkit-overflow-scrolling: touch; }
.gradio-wrapper iframe { height: 450px; }
```

- [ ] **Step 22: Test in DevTools mobile viewport**

- [ ] **Step 23: Commit**

```bash
git add public/index.html && git commit -m "feat: mobile layout polish"
```

---

## Chunk 4: Final Verification

### Task 6: Verify and Push

- [ ] **Step 24: Validate HTML**

```bash
python3 -c "
from html.parser import HTMLParser
class C(HTMLParser):
    def __init__(self):
        super().__init__(); self.errs=[]; self.stk=[]; self.void={'area','base','br','col','embed','hr','img','input','link','meta','param','source','track','wbr'}
    def handle_starttag(self,t,a):
        if t not in self.void: self.stk.append(t)
    def handle_endtag(self,t):
        if t in self.void: return
        if self.stk and self.stk[-1]==t: self.stk.pop()
        else: self.errs.append(f'</{t}>')
c=C(); c.feed(open('public/index.html').read())
print('OK' if not c.errs and not c.stk else f'Errors: {c.errs} Unclosed: {c.stk}')
"
```

- [ ] **Step 25: Check site loads**

```bash
curl -s -o /dev/null -w '%{http_code}' https://orbitalchaos.online
```

- [ ] **Step 26: Browser verification checklist**

1. Stats count up from 0 on scroll
2. Cards stagger in
3. Toggle buttons switch table values
4. SVG chart updates with toggle
5. Globe shows prediction arcs
6. Mobile layout correct (DevTools)

- [ ] **Step 27: Push**

```bash
git push origin main
```

---

## Session 2 Completion Checklist

- [ ] Scroll animations: fade, stagger, counters
- [ ] Toggle table: 4 conditions, dynamic table + SVG
- [ ] Prediction overlay: dashed arcs from predictions.json
- [ ] Mobile: touch-action, 2-col weather, scrollable toggles
- [ ] HTML validates
- [ ] All pushed

**After Session 1:** Replace `TBD` in `RESULTS_DATA` with real numbers from results JSONs.

**Session 3 can begin** — creates the prediction cron that writes real `predictions.json`.
