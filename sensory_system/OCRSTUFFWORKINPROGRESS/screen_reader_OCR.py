import sys
import mss
import numpy as np
from PIL import Image
import easyocr
import pyperclip
import Quartz

def get_screen_text(monitor_num=1, lang_list=['en'], region=None, use_gpu=False, as_lines=False):
    reader = easyocr.Reader(lang_list, gpu=use_gpu)
    with mss.mss() as sct:
        if region:
            monitor = region
        else:
            monitor = sct.monitors[monitor_num]
        sct_img = sct.grab(monitor)
        img = Image.frombytes('RGB', sct_img.size, sct_img.rgb)
        img_np = np.array(img)
        results = reader.readtext(img_np)
        if as_lines:
            return [r[1] for r in results]  # returns each line
        else:
            return " ".join([r[1] for r in results])
        
def get_clipboard_text(min_length=300):
    try:
        text = pyperclip.paste().strip()
        # Optionally: skip image data or very short/garbage
        if text and len(text) >= min_length:
            return text
    except Exception:
        pass
    return None

def get_active_window_region():
    platform = sys.platform
    if platform.startswith("darwin"):
        # macOS: Use Quartz (PyObjC)
        try:
            windows = Quartz.CGWindowListCopyWindowInfo(
                Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListOptionOnScreenAboveWindow,
                Quartz.kCGNullWindowID
            )
            for w in windows:
                if w.get('kCGWindowLayer', 0) == 0 and w.get('kCGWindowBounds'):
                    bounds = w['kCGWindowBounds']
                    return {
                        'top': int(bounds['Y']),
                        'left': int(bounds['X']),
                        'width': int(bounds['Width']),
                        'height': int(bounds['Height'])
                    }
        except Exception as e:
            print(f"Quartz failed: {e}")
            return None
    elif platform.startswith("win"):
        # Windows: Use pygetwindow
        try:
            import pygetwindow as gw
            win = gw.getActiveWindow()
            if win and hasattr(win, 'top'):
                return {'top': win.top, 'left': win.left, 'width': win.width, 'height': win.height}
        except Exception as e:
            print(f"pygetwindow failed: {e}")
            return None
    else:
        # Linux/other: Not implemented, fall back to full screen
        print("Active window detection not supported on this platform. Using full screen.")
        return None
   
def run_ocr_now(show_debug=True):
    clipboard_text = get_clipboard_text()
    if clipboard_text:
        if show_debug:
            print("[ARGUS] Detected text in clipboard:")
        text = clipboard_text
    else:
        region = get_active_window_region()
        if region:
            if show_debug:
                print("[ARGUS] Detected text in active window:")
            text = get_screen_text(region=region)
        else:
            if show_debug:
                print("[ARGUS] No active window detected. Running OCR on the whole screen:")
            text = get_screen_text(monitor_num=1)
    if show_debug:
        print("[ARGUS] Text:\n", text)
    return text


if __name__ == '__main__':
    print("Press Enter to trigger OCR (Ctrl+C to quit):")
    try:
        while True:
            input()  # Wait for user to press Enter (or hook up to a voice/hotkey/event)
            run_ocr_now(show_debug=True)
    except KeyboardInterrupt:
        print("Exiting ARGUS OCR system.")