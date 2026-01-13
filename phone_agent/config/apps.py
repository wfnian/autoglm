"""App name to package name mapping for supported applications."""

APP_PACKAGES: dict[str, str] = {
    # Social & Messaging
    "行家": "com.boc.tesip",
    "微信": "com.tencent.mm",
    "QQ": "com.tencent.mobileqq",
    "微博": "com.sina.weibo",

}


def get_package_name(app_name: str) -> str | None:
    """
    Get the package name for an app.

    Args:
        app_name: The display name of the app.

    Returns:
        The Android package name, or None if not found.
    """
    return APP_PACKAGES.get(app_name)


def get_app_name(package_name: str) -> str | None:
    """
    Get the app name from a package name.

    Args:
        package_name: The Android package name.

    Returns:
        The display name of the app, or None if not found.
    """
    for name, package in APP_PACKAGES.items():
        if package == package_name:
            return name
    return None


def list_supported_apps() -> list[str]:
    """
    Get a list of all supported app names.

    Returns:
        List of app names.
    """
    return list(APP_PACKAGES.keys())