Feature: Employee Management on OrangeHRM

  Scenario: Admin logs in and adds a new employee
    Given I am on "https://opensource-demo.orangehrmlive.com/web/index.php/auth/login"
    Then I should see the text "OrangeHRM"
    When I enter "Admin" in the username field
    And I enter "admin123" in the password field
    And I click the "Login" button
    Then I should see the text "Dashboard"