(() => {
  const frame = document.getElementById("case-study-frame");
  const loading = document.getElementById("case-loading");
  const imageDialog = document.getElementById("case-image-dialog");
  if (!frame || !imageDialog) return;

  imageDialog.querySelector("button").addEventListener("click", () => imageDialog.close());
  let resizeObserver;
  let dialogObserver;
  let scheduled = false;

  function connectCaseStudy() {
    let caseDocument;
    try {
      caseDocument = frame.contentDocument;
      if (!caseDocument || !caseDocument.querySelector(".page-shell")) return;
    } catch {
      // The GitHub Pages sites share an origin. Keep the frame usable if
      // this page is temporarily previewed from another origin.
      if (loading) loading.hidden = true;
      return;
    }

    resizeObserver?.disconnect();
    dialogObserver?.disconnect();
    const shell = caseDocument.querySelector(".page-shell");
    const embeddedStyle = caseDocument.createElement("style");
    embeddedStyle.textContent = `
      html { overflow: hidden; }
      body { background: white; }
      .page-shell { width: 100%; margin: 0; }
      .site-header, .page-shell > footer { display: none; }
      .section-frame { box-shadow: none; }
    `;
    caseDocument.head.appendChild(embeddedStyle);

    function resizeFrame() {
      if (scheduled) return;
      scheduled = true;
      requestAnimationFrame(() => {
        scheduled = false;
        const height = Math.ceil(shell.getBoundingClientRect().height) + 2;
        if (Math.abs(frame.getBoundingClientRect().height - height) > 1) {
          frame.style.height = `${height}px`;
        }
      });
    }

    resizeObserver = new ResizeObserver(resizeFrame);
    resizeObserver.observe(shell);
    resizeFrame();
    if (loading) loading.hidden = true;

    // Show enlarged evidence in the main viewport rather than halfway
    // down the tall embedded document.
    const embeddedDialog = caseDocument.getElementById("image-dialog");
    if (embeddedDialog) {
      dialogObserver = new MutationObserver(() => {
        if (!embeddedDialog.open) return;
        const sourceImage = embeddedDialog.querySelector("img");
        const image = imageDialog.querySelector("img");
        image.src = sourceImage.src;
        image.alt = sourceImage.alt || "Enlarged source evidence";
        imageDialog.querySelector("p").textContent = embeddedDialog.querySelector("p")?.textContent || "";
        embeddedDialog.close();
        if (!imageDialog.open) imageDialog.showModal();
      });
      dialogObserver.observe(embeddedDialog, { attributes: true, attributeFilter: ["open"] });
    }
  }

  frame.addEventListener("load", connectCaseStudy);
  connectCaseStudy();
})();
