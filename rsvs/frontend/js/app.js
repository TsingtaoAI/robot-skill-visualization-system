/**
 * 机器人技能可视化演示交互系统 V1.0 — 门户脚本
 * 演示服务在线检测、一键启动/停止、hostname 打开三维页
 */
(function () {
  const DEMOS = [
    { id: "skills", port: 8081, label: "技能演示" },
    { id: "nav", port: 8082, label: "导航演示" },
    { id: "play", port: 8083, label: "运动遥控" },
  ];
  const WAIT_MS = 180000;
  const POLL_MS = 2000;

  function demoHost() {
    const h = location.hostname || "127.0.0.1";
    return h === "" ? "127.0.0.1" : h;
  }

  function demoUrl(port) {
    const proto = location.protocol === "https:" ? "https:" : "http:";
    return `${proto}//${demoHost()}:${port}`;
  }

  function markActiveNav() {
    const path = (location.pathname || "/").replace(/\/+$/, "") || "/";
    document.querySelectorAll(".nav-links a[data-nav]").forEach((a) => {
      const key = a.getAttribute("data-nav");
      const map = {
        home: ["/", "/index.html"],
        skills: ["/skills.html"],
        nav: ["/nav.html"],
        play: ["/play.html"],
        help: ["/help.html"],
      };
      const hits = map[key] || [];
      if (hits.includes(path) || (key === "home" && path.endsWith("/frontend"))) {
        a.classList.add("active");
      }
    });
  }

  async function fetchStatus(demo) {
    try {
      const res = await fetch(`/api/status?id=${encodeURIComponent(demo.id)}`, {
        cache: "no-store",
      });
      if (!res.ok) return { up: false };
      return await res.json();
    } catch (_) {
      return { up: false };
    }
  }

  function setBadge(el, data) {
    if (!el) return;
    const up = Boolean(data && data.up);
    const starting = Boolean(data && data.starting);
    el.textContent = up ? "在线" : starting ? "启动中" : "未启动";
    el.classList.toggle("up", up);
    el.classList.toggle("down", !up && !starting);
    el.classList.toggle("starting", starting);
    const row = el.closest("li") || el.parentElement;
    const dot = row && row.querySelector(".dot");
    if (dot) {
      dot.classList.toggle("up", up);
      dot.classList.toggle("down", !up && !starting);
      dot.classList.toggle("starting", starting);
    }
  }

  function setLaunchBusy(id, busy, text) {
    document.querySelectorAll(`[data-launch-demo="${id}"]`).forEach((btn) => {
      btn.disabled = Boolean(busy);
      btn.classList.toggle("is-busy", Boolean(busy));
      if (text) btn.textContent = text;
    });
  }

  function defaultLaunchLabel(up) {
    return up ? "打开三维演示" : "启动并打开";
  }

  function syncLaunchButtons(id, up) {
    document.querySelectorAll(`[data-launch-demo="${id}"]`).forEach((btn) => {
      if (btn.classList.contains("is-busy")) return;
      btn.textContent = defaultLaunchLabel(up);
      btn.classList.toggle("is-online", up);
    });
    document.querySelectorAll(`[data-stop-demo="${id}"]`).forEach((btn) => {
      btn.disabled = !up;
      btn.removeAttribute("hidden");
    });
  }

  async function refreshStatus() {
    for (const demo of DEMOS) {
      const data = await fetchStatus(demo);
      document.querySelectorAll(`[data-status="${demo.id}"]`).forEach((badge) => {
        setBadge(badge, data);
      });
      syncLaunchButtons(demo.id, Boolean(data.up));
    }
  }

  function sleep(ms) {
    return new Promise((r) => setTimeout(r, ms));
  }

  function toast(msg) {
    let el = document.getElementById("hub-toast");
    if (!el) {
      el = document.createElement("div");
      el.id = "hub-toast";
      el.className = "hub-toast";
      document.body.appendChild(el);
    }
    el.textContent = msg;
    el.classList.add("show");
    clearTimeout(el._timer);
    el._timer = setTimeout(() => el.classList.remove("show"), 4500);
  }

  async function waitUntilUp(demo, deadline) {
    while (Date.now() < deadline) {
      const data = await fetchStatus(demo);
      document.querySelectorAll(`[data-status="${demo.id}"]`).forEach((badge) => {
        setBadge(badge, data);
      });
      if (data.up) return { ok: true };
      if (data.managed === false && data.starting === false) {
        await sleep(400);
        return { ok: false, crashed: true };
      }
      const left = Math.max(0, Math.ceil((deadline - Date.now()) / 1000));
      setLaunchBusy(demo.id, true, `等待就绪… ${left}s`);
      await sleep(POLL_MS);
    }
    return { ok: false, crashed: false };
  }

  async function launchAndOpen(demo) {
    setLaunchBusy(demo.id, true, "正在启动…");
    toast(`正在启动${demo.label}（首次加载仿真可能较慢）`);
    try {
      const res = await fetch("/api/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id: demo.id }),
      });
      const data = await res.json();
      if (!data.ok) {
        const detail = data.log_tail
          ? String(data.log_tail).trim().split("\n").slice(-3).join(" | ")
          : data.error || "启动失败";
        toast(detail.length > 160 ? detail.slice(0, 160) + "…" : detail);
        setLaunchBusy(demo.id, false, defaultLaunchLabel(false));
        return;
      }
      if (data.up) {
        window.open(demoUrl(demo.port), "_blank", "noopener");
        setLaunchBusy(demo.id, false, defaultLaunchLabel(true));
        refreshStatus();
        return;
      }
      const result = await waitUntilUp(demo, Date.now() + WAIT_MS);
      if (result.ok) {
        toast(`${demo.label}已就绪`);
        window.open(demoUrl(demo.port), "_blank", "noopener");
        setLaunchBusy(demo.id, false, defaultLaunchLabel(true));
      } else {
        let extra = "";
        try {
          const lr = await fetch(`/api/log?id=${encodeURIComponent(demo.id)}`, {
            cache: "no-store",
          });
          const ld = await lr.json();
          if (ld.log_tail) {
            const last = String(ld.log_tail).trim().split("\n").slice(-2).join(" ");
            extra = last ? `：${last}` : "";
          }
        } catch (_) {}
        if (!extra) {
          extra = result.crashed
            ? "：进程已退出，请查看 newtest/logs/"
            : `，请查看 newtest/logs/${demo.id}.log`;
        } else if (result.crashed) {
          extra = `（进程已退出）${extra}`;
        }
        toast(`${demo.label}启动失败${extra}`.slice(0, 220));
        setLaunchBusy(demo.id, false, defaultLaunchLabel(false));
      }
      refreshStatus();
    } catch (err) {
      toast("启动请求失败，请确认门户已用 Genesis 环境运行");
      setLaunchBusy(demo.id, false, defaultLaunchLabel(false));
    }
  }

  async function stopDemo(demo) {
    toast(`正在停止${demo.label}…`);
    try {
      const res = await fetch("/api/stop", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id: demo.id }),
      });
      const data = await res.json();
      toast(data.message || (data.ok ? "已发送停止" : "停止失败"));
      refreshStatus();
    } catch (_) {
      toast("停止请求失败");
    }
  }

  function bindLaunchButtons() {
    DEMOS.forEach((demo) => {
      document.querySelectorAll(`[data-launch-demo="${demo.id}"]`).forEach((btn) => {
        btn.addEventListener("click", (e) => {
          e.preventDefault();
          if (btn.disabled || btn.classList.contains("is-busy")) return;
          launchAndOpen(demo);
        });
      });
      document.querySelectorAll(`[data-stop-demo="${demo.id}"]`).forEach((btn) => {
        btn.addEventListener("click", (e) => {
          e.preventDefault();
          if (btn.disabled) return;
          stopDemo(demo);
        });
      });
    });
  }

  document.addEventListener("DOMContentLoaded", () => {
    markActiveNav();
    bindLaunchButtons();
    refreshStatus();
    setInterval(refreshStatus, 4000);
  });
})();
