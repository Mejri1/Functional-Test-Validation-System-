Feature: Employee Management on OrangeHRM

  Scenario: Admin adds a new employee with profile picture
    Given I am on "https://opensource-demo.orangehrmlive.com/web/index.php/auth/login"
    Then I should see the text "Login"
    When I enter "Admin" in the username field
    And I enter "admin123" in the password field
    And I click the "Login" button
    Then I should see the text "Dashboard"
    When I click the "PIM" menu item
    Then I should see the text "Employee List"
    When I click the  "Add Employee" button
    Then I should see the text "Add Employee"
    When I enter "John" in the first name field
    And I enter "Michael" in the middle name field
    And I enter "Doe" in the last name field
    When I upload the file "C:\Users\MSI\Downloads\8815077.png" to the profile picture input field
    And I click the "Save" button
    Then I should see the text "Personal Details"
    And I should see the text "John"