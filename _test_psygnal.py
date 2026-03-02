"""Quick test to check psygnal compilation state."""
import psygnal
print("version:", psygnal.__version__)
print("SignalInstance type:", type(psygnal.SignalInstance))

# Try subclassing
try:
    class TestSub(psygnal.SignalInstance): pass
    print("Subclassing SignalInstance: OK")
except TypeError as e:
    print("Subclassing SignalInstance FAILED:", e)

# Try decompile
try:
    psygnal.utils.decompile()
    class TestSub2(psygnal.SignalInstance): pass
    print("After decompile, subclassing SignalInstance: OK")
except Exception as e:
    print("After decompile:", e)

# Test direct instantiation (composition approach)
try:
    inst = psygnal.SignalInstance((int,), name="test")
    print("Direct instantiation: OK, name =", inst.name)
except Exception as e:
    print("Direct instantiation FAILED:", e)
