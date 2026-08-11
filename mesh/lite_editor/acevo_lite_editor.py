#!/usr/bin/env python3
"""
AC EVO Lite Editor - minimal standalone reader/editor for .actor and
.carlightingsystem files.

No server, no third-party dependencies: tkinter (stdlib) + acevo_lite_codec.
The UI and the codec are kept in separate files on purpose, so the codec can
be reused as-is inside a future Blender add-on (bpy panel) without dragging
this tkinter UI along.

Usage:
    python acevo_lite_editor.py [file.actor|file.carlightingsystem]
"""

import json
import shutil
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext

sys.path.insert(0, str(Path(__file__).resolve().parent))
import acevo_lite_codec as codec

SUPPORTED_EXTS = (".actor", ".carlightingsystem")
FILETYPES = [
    ("Supported files", "*.actor *.carlightingsystem"),
    ("Actor files", "*.actor"),
    ("Car lighting system files", "*.carlightingsystem"),
]


def backup_once(path: Path) -> bool:
    """Create '<name>.<ext>.bak' next to path, but only if it doesn't exist yet,
    so the very first pre-edit version of a session is always preserved."""
    bak = path.with_suffix(path.suffix + ".bak")
    if not bak.exists():
        shutil.copy2(path, bak)
        return True
    return False


class LiteEditor(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AC EVO Lite Editor")
        self.geometry("900x650")
        self.current_path = None
        self.dirty = False

        self.text = scrolledtext.ScrolledText(self, wrap="none", undo=True, font=("Consolas", 10))
        self.text.pack(fill="both", expand=True)
        self.text.bind("<<Modified>>", self._on_modified)
        self.text.tag_configure("find_match", background="#ffe38a")
        self.text.tag_configure("find_current", background="#ff9d3d")

        self.status = tk.StringVar(value="No file loaded - File > Open (.actor / .carlightingsystem)")
        tk.Label(self, textvariable=self.status, anchor="w", relief="sunken").pack(fill="x")

        self.find_var = tk.StringVar()
        self.find_var.trace_add("write", lambda *a: self._highlight_all(self.find_var.get()))
        self.find_count = tk.StringVar()
        self._build_find_bar()

        self._build_menu()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_menu(self):
        menubar = tk.Menu(self)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Open...", command=self.open_file, accelerator="Ctrl+O")
        filemenu.add_command(label="Save", command=self.save_file, accelerator="Ctrl+S")
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=self._on_close)
        menubar.add_cascade(label="File", menu=filemenu)

        editmenu = tk.Menu(menubar, tearoff=0)
        editmenu.add_command(label="Find...", command=self.show_find, accelerator="Ctrl+F")
        editmenu.add_command(label="Find Next", command=lambda: self.find_next(), accelerator="F3")
        editmenu.add_command(label="Find Previous", command=lambda: self.find_next(reverse=True),
                              accelerator="Shift+F3")
        menubar.add_cascade(label="Edit", menu=editmenu)

        self.config(menu=menubar)
        self.bind_all("<Control-o>", lambda e: self.open_file())
        self.bind_all("<Control-s>", lambda e: self.save_file())
        self.bind_all("<Control-f>", lambda e: self.show_find())
        self.bind_all("<F3>", lambda e: self.find_next())
        self.bind_all("<Shift-F3>", lambda e: self.find_next(reverse=True))

    def _build_find_bar(self):
        self.find_frame = tk.Frame(self, relief="raised", bd=1)
        tk.Label(self.find_frame, text="Find:").pack(side="left", padx=(6, 2), pady=3)
        entry = tk.Entry(self.find_frame, textvariable=self.find_var, width=40)
        entry.pack(side="left", padx=2, pady=3)
        entry.bind("<Return>", lambda e: self.find_next())
        entry.bind("<Shift-Return>", lambda e: self.find_next(reverse=True))
        entry.bind("<Escape>", lambda e: self.hide_find())
        self.find_entry = entry
        tk.Button(self.find_frame, text="Next", command=lambda: self.find_next()).pack(side="left", padx=2)
        tk.Button(self.find_frame, text="Previous", command=lambda: self.find_next(reverse=True)).pack(
            side="left", padx=2)
        tk.Label(self.find_frame, textvariable=self.find_count, fg="gray30").pack(side="left", padx=6)
        tk.Button(self.find_frame, text="✕", command=self.hide_find, relief="flat", padx=6).pack(
            side="right", padx=4)
        # Built once at startup but only packed (shown) on demand by show_find().

    def show_find(self):
        if not self.find_frame.winfo_ismapped():
            self.find_frame.pack(side="top", fill="x", before=self.text)
        self.find_entry.focus_set()
        self.find_entry.select_range(0, "end")
        if self.find_var.get():
            self._highlight_all(self.find_var.get())

    def hide_find(self):
        self.find_frame.pack_forget()
        self.text.tag_remove("find_match", "1.0", "end")
        self.text.tag_remove("find_current", "1.0", "end")
        self.text.focus_set()

    def _highlight_all(self, pattern: str):
        self.text.tag_remove("find_match", "1.0", "end")
        if not pattern:
            self.find_count.set("")
            return
        count = 0
        start = "1.0"
        while True:
            idx = self.text.search(pattern, start, stopindex="end", nocase=True)
            if not idx:
                break
            end_idx = f"{idx}+{len(pattern)}c"
            self.text.tag_add("find_match", idx, end_idx)
            count += 1
            start = end_idx
        self.find_count.set(f"{count} match{'es' if count != 1 else ''}" if count else "Not found")

    def find_next(self, reverse: bool = False):
        pattern = self.find_var.get()
        if not pattern:
            self.show_find()
            return
        self.text.tag_remove("find_current", "1.0", "end")
        start = self.text.index("insert")
        if reverse:
            idx = self.text.search(pattern, start, backwards=True, stopindex="1.0", nocase=True)
            if not idx:
                idx = self.text.search(pattern, "end", backwards=True, stopindex=start, nocase=True)
        else:
            idx = self.text.search(pattern, start, stopindex="end", nocase=True)
            if not idx:
                idx = self.text.search(pattern, "1.0", stopindex=start, nocase=True)
        if not idx:
            self.find_count.set("Not found")
            return
        end_idx = f"{idx}+{len(pattern)}c"
        self.text.tag_add("find_current", idx, end_idx)
        self.text.mark_set("insert", idx if reverse else end_idx)
        self.text.see(idx)
        self._highlight_all(pattern)

    def _on_modified(self, event=None):
        if self.text.edit_modified():
            self.dirty = True
            self._update_title()
            self.text.edit_modified(False)

    def _update_title(self):
        name = self.current_path.name if self.current_path else "no file"
        star = "*" if self.dirty else ""
        self.title(f"AC EVO Lite Editor - {name}{star}")

    def _confirm_discard(self):
        if not self.dirty:
            return True
        return messagebox.askyesno("Unsaved changes", "Discard unsaved changes?")

    def open_file(self):
        if not self._confirm_discard():
            return
        chosen = filedialog.askopenfilename(title="Open .actor / .carlightingsystem", filetypes=FILETYPES)
        if chosen:
            self.load_path(Path(chosen))

    def load_path(self, path: Path):
        if path.suffix.lower() not in SUPPORTED_EXTS:
            messagebox.showerror("Unsupported file", "Only .actor and .carlightingsystem files are supported.")
            return
        try:
            raw = path.read_bytes()
            tree = codec.decode_message(raw)
            roundtrip_ok = codec.encode_message(tree) == raw
        except Exception as e:
            messagebox.showerror("Decode failed", str(e))
            return

        text = json.dumps(tree, indent=2, ensure_ascii=False)
        self.text.delete("1.0", "end")
        self.text.insert("1.0", text)
        self.text.edit_modified(False)
        self.dirty = False
        self.current_path = path
        self._update_title()

        rt_msg = "round-trip OK" if roundtrip_ok else "WARNING: round-trip mismatch on load"
        self.status.set(f"{path} - {len(raw)} bytes - {rt_msg}")
        if not roundtrip_ok:
            messagebox.showwarning(
                "Round-trip warning",
                "This file did not re-encode to identical bytes on load.\n"
                "Editing and saving may not preserve unrecognized data exactly.")

    def save_file(self):
        if self.current_path is None:
            messagebox.showinfo("No file", "Open a file first.")
            return
        text = self.text.get("1.0", "end-1c")
        try:
            tree = json.loads(text)
        except json.JSONDecodeError as e:
            messagebox.showerror("Invalid JSON", str(e))
            return
        try:
            data = codec.encode_message(tree)
        except Exception as e:
            messagebox.showerror("Encode failed", str(e))
            return
        try:
            made_backup = backup_once(self.current_path)
            self.current_path.write_bytes(data)
        except OSError as e:
            messagebox.showerror("Save failed", str(e))
            return

        self.dirty = False
        self.text.edit_modified(False)
        self._update_title()
        note = " (backup created)" if made_backup else ""
        self.status.set(f"Saved {self.current_path} - {len(data)} bytes{note}")

    def _on_close(self):
        if self._confirm_discard():
            self.destroy()


def main():
    app = LiteEditor()
    if len(sys.argv) > 1:
        arg_path = Path(sys.argv[1])
        if arg_path.exists():
            app.after(10, lambda: app.load_path(arg_path))
    app.mainloop()


if __name__ == "__main__":
    main()
