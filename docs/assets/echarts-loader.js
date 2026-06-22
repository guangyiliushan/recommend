/**
 * ECharts loader — 自动渲染页面中所有 .echarts div。
 *
 * 每个 <div class="echarts"> 需有 data-src 指向 .echarts.json 文件。
 */
(function () {
  "use strict";

  function renderCharts() {
    var containers = document.querySelectorAll(".echarts");
    containers.forEach(function (el) {
      var src = el.getAttribute("data-src");
      if (!src) return;

      var height = el.getAttribute("data-height") || "400px";
      if (!el.style.minHeight) {
        el.style.width = "100%";
        el.style.minHeight = height;
        el.style.height = height;
      }

      if (el.getAttribute("data-rendered") === "true") return;
      el.setAttribute("data-rendered", "true");

      fetch(src)
        .then(function (resp) {
          if (!resp.ok) throw new Error("HTTP " + resp.status);
          return resp.json();
        })
        .then(function (json) {
          var option = json.echarts_option || json;
          var chart = echarts.init(el);
          chart.setOption(option);
          window.addEventListener("resize", function () {
            chart.resize();
          });
        })
        .catch(function (err) {
          el.textContent = "[Chart unavailable: " + err.message + "]";
          el.style.height = "auto";
          el.style.minHeight = "auto";
          el.style.padding = "1rem";
          el.style.color = "#999";
        });
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", renderCharts);
  } else {
    renderCharts();
  }

  // MkDocs instant navigation: re-render on page transition
  if (typeof document$ !== "undefined") {
    document$.subscribe(function () {
      setTimeout(renderCharts, 50);
    });
  }
})();
