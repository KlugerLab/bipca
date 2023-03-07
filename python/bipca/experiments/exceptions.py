from typing import Dict
import traceback


def handle_multiple_exceptions(
    parent_exception_type: type, msg: str, exceptions: Dict[str, Exception]
):
    if len(exceptions) > 1:
        blame_list = []
        exc_list = []
        for blame_str, exc in exceptions.items():
            blame_list.append(blame_str)
            tb = traceback.format_exception(None, exc, exc.__traceback__)
            tb.insert(0, f'\n------Start {blame_str}--------\n ')
            tb.append(f'------End {blame_str}--------\n ')
            exc_list.append(''.join(tb))
        msg += ', '.join(blame_list) + '.\nTracebacks:' + ''.join(exc_list)

        raise parent_exception_type(msg)
    elif len(exceptions) == 1:
        to_blame = list(exceptions.keys())[0]
        exc = exceptions[to_blame]
        msg += to_blame
        raise parent_exception_type(msg) from exc
    