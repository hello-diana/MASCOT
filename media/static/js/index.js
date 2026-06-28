/* ===========================================================
   MASCOT — interactivity (vanilla JS)
   - figure lightbox / zoom
   - persona detail modals
   - sticky nav scroll-spy + mobile toggle
   - reveal-on-scroll
   - copy BibTeX
   =========================================================== */

(function () {
  "use strict";

  /* ---------- helpers ---------- */
  function lockScroll(lock) {
    document.body.style.overflow = lock ? "hidden" : "";
  }

  /* ---------- 1. Figure lightbox ---------- */
  function initLightbox() {
    var lb = document.getElementById("lightbox");
    if (!lb) return;
    var img = lb.querySelector("img");
    var cap = lb.querySelector(".lb-caption");

    function open(src, caption) {
      img.src = src;
      img.alt = caption || "";
      cap.textContent = caption || "";
      lb.classList.add("is-open");
      lockScroll(true);
    }
    function close() {
      lb.classList.remove("is-open");
      lockScroll(false);
    }

    document.querySelectorAll(".zoomable").forEach(function (el) {
      el.addEventListener("click", function () {
        var caption = el.getAttribute("data-caption") ||
                      el.getAttribute("alt") || "";
        open(el.getAttribute("src"), caption);
      });
    });

    lb.addEventListener("click", function (e) {
      if (e.target === lb || e.target.classList.contains("overlay-close")) close();
    });
    document.addEventListener("keydown", function (e) {
      if (e.key === "Escape" && lb.classList.contains("is-open")) close();
    });
  }

  /* ---------- 2. Persona modals ---------- */
  function initPersonaModals() {
    var modal = document.getElementById("persona-modal");
    if (!modal) return;
    var box = modal.querySelector(".modal-card-box");
    var emoji = modal.querySelector(".modal-emoji");
    var name = modal.querySelector(".modal-name");
    var role = modal.querySelector(".modal-role");
    var traitsWrap = modal.querySelector(".modal-traits");
    var example = modal.querySelector(".modal-example");

    function open(card) {
      emoji.textContent = card.getAttribute("data-emoji") || "🤝";
      name.textContent = card.getAttribute("data-name") || "";
      role.textContent = card.getAttribute("data-role") || "";
      var accent = card.getAttribute("data-accent") || "var(--primary-1)";
      box.style.borderTopColor = accent;

      traitsWrap.innerHTML = "";
      (card.getAttribute("data-traits") || "").split(",").forEach(function (t) {
        t = t.trim();
        if (!t) return;
        var chip = document.createElement("span");
        chip.className = "trait-chip";
        chip.textContent = t;
        traitsWrap.appendChild(chip);
      });

      example.innerHTML = card.getAttribute("data-example") || "";
      modal.classList.add("is-open");
      lockScroll(true);
    }
    function close() {
      modal.classList.remove("is-open");
      lockScroll(false);
    }

    document.querySelectorAll(".persona-card").forEach(function (card) {
      card.addEventListener("click", function () { open(card); });
    });
    modal.addEventListener("click", function (e) {
      if (e.target === modal || e.target.classList.contains("overlay-close")) close();
    });
    document.addEventListener("keydown", function (e) {
      if (e.key === "Escape" && modal.classList.contains("is-open")) close();
    });
  }

  /* ---------- 3. Sticky nav: mobile toggle + scroll-spy ---------- */
  function initNav() {
    var toggle = document.querySelector(".nav-toggle");
    var links = document.querySelector(".topnav .links");
    if (toggle && links) {
      toggle.addEventListener("click", function () {
        links.classList.toggle("is-open");
        var open = links.classList.contains("is-open");
        toggle.setAttribute("aria-expanded", open ? "true" : "false");
      });
      links.querySelectorAll("a").forEach(function (a) {
        a.addEventListener("click", function () { links.classList.remove("is-open"); });
      });
    }

    var navLinks = Array.prototype.slice.call(
      document.querySelectorAll('.topnav .links a[href^="#"]')
    );
    if (!navLinks.length || !("IntersectionObserver" in window)) return;

    var map = {};
    navLinks.forEach(function (a) {
      var id = a.getAttribute("href").slice(1);
      var sec = document.getElementById(id);
      if (sec) map[id] = a;
    });

    var observer = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) {
          navLinks.forEach(function (a) { a.classList.remove("is-active"); });
          var active = map[entry.target.id];
          if (active) active.classList.add("is-active");
        }
      });
    }, { rootMargin: "-35% 0px -55% 0px", threshold: 0 });

    Object.keys(map).forEach(function (id) {
      observer.observe(document.getElementById(id));
    });
  }

  /* ---------- 4. Reveal on scroll ---------- */
  function initReveal() {
    var els = document.querySelectorAll(".reveal");
    if (!els.length) return;
    if (!("IntersectionObserver" in window)) {
      els.forEach(function (el) { el.classList.add("in-view"); });
      return;
    }
    var observer = new IntersectionObserver(function (entries, obs) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) {
          entry.target.classList.add("in-view");
          obs.unobserve(entry.target);
        }
      });
    }, { threshold: 0.08, rootMargin: "0px 0px -8% 0px" });
    els.forEach(function (el) { observer.observe(el); });
  }

  /* ---------- 5. Copy BibTeX ---------- */
  function initCopyBib() {
    var btn = document.querySelector(".copy-bib");
    if (!btn) return;
    var pre = document.querySelector(".bibtex-wrap pre code") ||
              document.querySelector(".bibtex-wrap pre");
    btn.addEventListener("click", function () {
      var text = pre ? pre.innerText : "";
      var done = function () {
        btn.classList.add("is-copied");
        btn.innerHTML = '<i class="fas fa-check"></i> Copied';
        setTimeout(function () {
          btn.classList.remove("is-copied");
          btn.innerHTML = '<i class="fas fa-copy"></i> Copy';
        }, 1800);
      };
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(done).catch(function () { fallback(text, done); });
      } else {
        fallback(text, done);
      }
    });
    function fallback(text, done) {
      var ta = document.createElement("textarea");
      ta.value = text;
      ta.style.position = "fixed";
      ta.style.opacity = "0";
      document.body.appendChild(ta);
      ta.select();
      try { document.execCommand("copy"); } catch (e) {}
      document.body.removeChild(ta);
      done(); // always show feedback, even if the copy command was blocked
    }
  }

  /* ---------- boot ---------- */
  document.addEventListener("DOMContentLoaded", function () {
    initLightbox();
    initPersonaModals();
    initNav();
    initReveal();
    initCopyBib();
  });
})();
