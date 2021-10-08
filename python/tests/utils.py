from functools import wraps
import contextlib
import io
import sys


def raises(exc, someopt=None):
	def outer(fn):
		@wraps(fn)
		def wrapper(*args, **kwargs):
			try:
				fn(*args,**kwargs)
			except Exception as err:
				if isinstance(err,exc):
					return True
				else:
					raise err
			return False
		return wrapper
	return outer

@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stderr
    sys.stdout = io.BytesIO()
    yield
    sys.stdout = save_stderr