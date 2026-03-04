Feature: User Authentication on AutomationExercise

  Scenario: Login with incorrect email and password
    Given I am on "http://automationexercise.com"
    When I click the "Signup / Login" button
    Then I should see the text "Login to your account"
    When I enter "wrongemail@test.com" in the email field
    And I enter "wrongpassword123" in the password field
    And I click the "Login" button
    Then I should see the text "Your email or password is incorrect!"