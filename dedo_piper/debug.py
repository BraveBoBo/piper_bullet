import debugpy
# vscode新版本通过debugpy代替ptvsd，才能对外部包的代码打断点。
import os
def enable_remote_debug(port = None):
    """远程调试代码

    代码主动断点: debugpy.breakpoint()

    Args:
        port (int, optional): 调试端口. Defaults to None.
    """
    try:
        import debugpy
        if port is None:
            ENV_DEBUG_PORT = os.environ.get('DEBUG_PORT')
            port = int(ENV_DEBUG_PORT) if ENV_DEBUG_PORT else 5678
            if not ENV_DEBUG_PORT:
                print('Set env DEBUG_PORT can change default port. (Linux example cmd: export DEBUG_PORT=5678)')
        address = ('0.0.0.0', port)
        debugpy.listen(address)
        print('### Wait Remote Debug (port:' + str(port) + ') ###')
        debugpy.wait_for_client()
        print('### Connected Remote Debug ###')
    except BaseException as e:
        print('enable_remote_debug err:', e)
        return False
    else:
        return True