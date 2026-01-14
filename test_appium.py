from appium import webdriver
from appium.options.android import UiAutomator2Options

# 配置 Options
options = UiAutomator2Options()
options.platformName = "Android"
options.automationName = "UiAutomator2"
options.deviceName = "d3523133"   # 用 adb devices 查到的真机 ID
options.appPackage = "com.android.calculator2"
options.appActivity = ".CalculatorActivity"

# 连接 Appium Server
driver = webdriver.Remote("http://127.0.0.1:4723", options=options)

# 示例操作
driver.find_element("id", "com.android.calculator2:id/digit_2").click()
driver.find_element("id", "com.android.calculator2:id/op_add").click()
driver.find_element("id", "com.android.calculator2:id/digit_3").click()
driver.find_element("id", "com.android.calculator2:id/eq").click()

print("结果:", driver.find_element("id", "com.android.calculator2:id/result").text)

driver.quit()
