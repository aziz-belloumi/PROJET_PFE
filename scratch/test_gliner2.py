import sys
import traceback

try:
    from gliner import GLiNER

    # --- Monkey-patch for GLiNER + new huggingface_hub ---
    if hasattr(GLiNER, '_from_pretrained'):
        _orig_from_pretrained = GLiNER._from_pretrained
        @classmethod
        def _patched_from_pretrained(cls, *args, **kwargs):
            kwargs.setdefault('proxies', None)
            kwargs.setdefault('resume_download', False)
            return _orig_from_pretrained.__func__(cls, *args, **kwargs)
        GLiNER._from_pretrained = _patched_from_pretrained

    print("GLiNER version:", getattr(GLiNER, '__version__', 'unknown'))
    model = GLiNER.from_pretrained("urchade/gliner_multi-v2.1")
    print("Successfully loaded model!")
except Exception as e:
    print("FAILED TO LOAD:")
    traceback.print_exc()
