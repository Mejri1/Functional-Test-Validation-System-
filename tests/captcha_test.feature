Feature: Captcha Solving on 2Captcha Demo

  Scenario: User solves a normal image captcha
    Given I am on "https://2captcha.com/demo/normal"
    Then I should see the text "Normal Captcha demo"
    When I solve the captcha challenge on the page
    And I click the "Check" button
    Then I should see the text "Captcha is passed successfully"