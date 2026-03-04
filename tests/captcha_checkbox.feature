Feature: Checkbox Captcha on 2Captcha Demo

  Scenario: Browser Use solves a reCAPTCHA v2 checkbox
    Given I am on "https://2captcha.com/demo/recaptcha-v2"
    Then I should see the text "reCAPTCHA V2"
    When I solve the captcha challenge on the page
    And I click the "Check" button
    Then I should see the text "Captcha is passed successfully"