Feature: User Authentication on AutomationExercise

  Scenario: Login with incorrect email and password
    Given I launch the browser
    And I navigate to "http://automationexercise.com"
    Then I should see the home page
    When I click the "Signup / Login" button
    Then I should see the text "Login to your account"
    When I enter "wrongemail@test.com" in the email field
    And I enter "wrongpassword123" in the password field
    And I click the "Login" button
    Then I should see the error message "Your email or password is incorrect!"