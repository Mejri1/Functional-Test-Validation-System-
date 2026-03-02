Feature: Employee Management on OrangeHRM

  Scenario: Admin logs in and adds a new employee
    Given I am on "https://opensource-demo.orangehrmlive.com/web/index.php/auth/login"
    Then I should see the text "OrangeHRM"
    When I enter "Admin" in the username field
    And I enter "admin123" in the password field
    And I click the "Login" button
    Then I should see the text "Dashboard"
    When I click the "PIM" menu item
    Then I should see the text "Employee List"
    When I click the "Add Employee" button
    Then I should see the text "Add Employee"
    When I enter "John" in the first name field
    And I enter "Doe" in the last name field
    And I click the "Save" button
    Then I should see the text "Personal Details"