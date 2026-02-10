#!/usr/bin/env python3
"""
multi_launcher_simple.py

Minimal, clean launcher: shows five upload slots (one per view).
No console area. No blocking popups. Status updates are inline.
Window is raised to the front on startup.

Model <-> script mapping (auto-selected; no model upload UI):
 - Side view  : score.py        --model side_view_model.pt
 - Rear view  : scale.py        --model rear_view_model.pt
 - Top view   : 1.py            --model top_view_model.pt
 - Udder side : udder.py        --model udder.pt
 - Udder view : udder_view.py   --model udder_view_model.pt
"""
import os
import sys
import threading
import subprocess
import shlex
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import time

PANELS = [
    {"name": "Side view",  "script": "score.py",       "model": "side_view_model.pt"},
    {"name": "Rear view",  "script": "scale.py",       "model": "rear_view_model.pt"},
    {"name": "Top view",   "script": "1.py",           "model": "top_view_model.pt"},
    {"name": "Udder side", "script": "udder.py",       "model": "udder.pt"},
    {"name": "Udder view", "script": "udder_view.py",  "model": "udder_view_model.pt"},
]

IMG_EXTS = [("Images", "*.jpg;*.jpeg;*.png;*.bmp;*.tiff"), ("All files", "*")]


def find_script_path(script_name):
    """Return absolute path to script if found in cwd or same dir as this file, else None."""
    if os.path.isabs(script_name) and os.path.exists(script_name):
        return script_name
    c = os.path.join(os.getcwd(), script_name)
    if os.path.exists(c):
        return os.path.abspath(c)
    # try script next to this launcher
    try:
        base_dir = os.path.dirname(__file__)
    except NameError:
        base_dir = os.getcwd()
    c2 = os.path.join(base_dir, script_name)
    if os.path.exists(c2):
        return os.path.abspath(c2)
    return None


class SimpleLauncher(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Minimal Pipeline Launcher")
        self.minsize(880, 520)
        self.selected = {i: None for i in range(len(PANELS))}
        self.preview_imgs = {}
        self._build_ui()
        # bring to front
        self.lift()
        self.attributes("-topmost", True)
        self.after(200, lambda: self.attributes("-topmost", False))

    def _build_ui(self):
        container = ttk.Frame(self, padding=12)
        container.pack(fill="both", expand=True)

        header = ttk.Label(container, text="Upload images for each view", font=("Segoe UI", 14, "bold"))
        header.pack(anchor="nw")

        grid = ttk.Frame(container)
        grid.pack(fill="both", expand=True, pady=(8,0))

        # create a 2-column layout for neatness (3 rows: 2+2+1)
        cols = 2
        for i, panel in enumerate(PANELS):
            r = i // cols
            c = i % cols
            frame = ttk.LabelFrame(grid, text=panel["name"], padding=8)
            frame.grid(row=r, column=c, padx=8, pady=8, sticky="nwe")
            frame.columnconfigure(1, weight=1)

            # preview canvas
            canvas = tk.Canvas(frame, width=260, height=170, bg="black", highlightthickness=0)
            canvas.grid(row=0, column=0, rowspan=3, sticky="w", padx=(0,8))

            lbl = ttk.Label(frame, text="No image selected", wraplength=300)
            lbl.grid(row=0, column=1, sticky="nw")

            btn_upload = ttk.Button(frame, text="Upload Image", command=lambda idx=i: self._upload_image(idx))
            btn_upload.grid(row=1, column=1, sticky="we", pady=(8,4))

            # Run button kept but small and unobtrusive; disabled if script missing
            btn_run = ttk.Button(frame, text="Run", width=10, command=lambda idx=i: self._run_panel(idx))
            btn_run.grid(row=2, column=1, sticky="w")

            status = ttk.Label(frame, text="", foreground="#444444")
            status.grid(row=3, column=0, columnspan=2, sticky="w", pady=(6,0))

            # store refs
            frame._refs = {
                "canvas": canvas, "label": lbl, "upload": btn_upload, "run": btn_run, "status": status
            }

            # disable run if script not found
            script_path = find_script_path(panel["script"])
            if script_path is None:
                btn_run.state(["disabled"])
                status.config(text=f"Script not found: {panel['script']}")
            else:
                status.config(text=f"Model: {panel['model']}")

            self.preview_imgs[i] = None

        # final row with minimal controls (Run All)
        controls = ttk.Frame(container)
        controls.pack(fill="x", pady=(6,0))
        run_all = ttk.Button(controls, text="Run All (selected images)", command=self._run_all)
        run_all.pack(side="left")
        spacer = ttk.Label(controls, text="")
        spacer.pack(side="left", padx=8)
        hint = ttk.Label(controls, text="Select images, then click Run for each view or Run All.", foreground="#555555")
        hint.pack(side="left")

    def _upload_image(self, idx):
        path = filedialog.askopenfilename(title=f"Select image for {PANELS[idx]['name']}", filetypes=IMG_EXTS)
        if not path:
            return
        # store and show
        self.selected[idx] = path
        lbl = self._get_ref(idx, "label")
        lbl.config(text=os.path.basename(path))
        self._show_preview(idx, path)
        # enable run button if script exists
        btn_run = self._get_ref(idx, "run")
        script_path = find_script_path(PANELS[idx]["script"])
        if script_path:
            btn_run.state(["!disabled"])

    def _show_preview(self, idx, path):
        try:
            im = Image.open(path)
            im.thumbnail((260, 170), Image.LANCZOS)
            tkimg = ImageTk.PhotoImage(im)
            self.preview_imgs[idx] = tkimg
            canvas = self._get_ref(idx, "canvas")
            canvas.delete("all")
            canvas.create_image(0, 0, anchor="nw", image=tkimg)
            # clear status text
            self._get_ref(idx, "status").config(text=f"Model: {PANELS[idx]['model']}")
        except Exception as e:
            self._get_ref(idx, "status").config(text=f"Preview failed: {e}")

    def _get_ref(self, idx, name):
        # navigate UI tree to find stored refs
        grid = self.winfo_children()[0].winfo_children()[1]  # container -> grid frame
        # compute position: each child is a LabelFrame in grid order
        # simpler: iterate LabelFrame children and pick idx
        frames = [c for c in grid.winfo_children() if isinstance(c, ttk.LabelFrame)]
        frame = frames[idx]
        return frame._refs[name]

    def _run_panel(self, idx):
        img = self.selected.get(idx)
        if not img:
            # update inline status (no popups)
            self._get_ref(idx, "status").config(text="No image selected")
            return
        panel = PANELS[idx]
        script_path = find_script_path(panel["script"])
        if not script_path:
            self._get_ref(idx, "status").config(text=f"Script missing: {panel['script']}")
            return

        # run in background to avoid UI freeze; minimal inline status updates
        def worker():
            status = self._get_ref(idx, "status")
            status.config(text="Running...")
            cmd = [sys.executable, script_path, "--model", panel["model"], "--image", img]
            try:
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
                # consume output but do not show console; keep last line as status
                last = ""
                for line in proc.stdout:
                    last = line.strip()
                proc.wait()
                if proc.returncode == 0:
                    status.config(text="Finished successfully")
                else:
                    status.config(text=f"Failed (exit {proc.returncode})")
            except Exception as e:
                status.config(text=f"Run error: {e}")

        t = threading.Thread(target=worker, daemon=True)
        t.start()

    def _run_all(self):
        # sequentially run panels that have images selected
        def seq():
            for i, panel in enumerate(PANELS):
                if not self.selected.get(i):
                    self._get_ref(i, "status").config(text="Skipped (no image)")
                    continue
                self._get_ref(i, "status").config(text="Running...")
                script_path = find_script_path(panel["script"])
                if not script_path:
                    self._get_ref(i, "status").config(text=f"Script missing: {panel['script']}")
                    continue
                cmd = [sys.executable, script_path, "--model", panel["model"], "--image", self.selected[i]]
                try:
                    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
                    last = ""
                    for line in proc.stdout:
                        last = line.strip()
                    proc.wait()
                    if proc.returncode == 0:
                        self._get_ref(i, "status").config(text="Finished")
                    else:
                        self._get_ref(i, "status").config(text=f"Failed (exit {proc.returncode})")
                except Exception as e:
                    self._get_ref(i, "status").config(text=f"Run error: {e}")
            # small global hint after finishing
            time.sleep(0.2)

        threading.Thread(target=seq, daemon=True).start()


def main():
    app = SimpleLauncher()
    app.lift()
    app.attributes("-topmost", True)
    app.after(200, lambda: app.attributes("-topmost", False))
    app.mainloop()


if __name__ == "__main__":
    main()
