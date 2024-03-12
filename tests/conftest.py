def skip_s4_test():
    """
     Utility function for skipping the test if S4 is not installed,

    Returns: True if S4 not installed False otherwise
    """

    try:
        import S4
        return False
    except ModuleNotFoundError:
        return True