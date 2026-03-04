Feature: Image Select Captcha on 2Captcha Demo

  Scenario: Browser Use solves a click image captcha
    Given I am on "https://2captcha.com/demo/clickcaptcha"
    Then I should see the text "Click Captcha demo"
    When I solve the captcha challenge on the page
    And I click the "Check" button
    Then I should see the text "Captcha is passed successfully"