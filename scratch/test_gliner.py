import sys
import traceback

try:
    from gliner import GLiNER
    print("GLiNER version:", getattr(GLiNER, '__version__', 'unknown'))
    model = GLiNER.from_pretrained("urchade/gliner_multi-v2.1")
    print("Successfully loaded model!")
except Exception as e:
    print("FAILED TO LOAD:")
    traceback.print_exc()
