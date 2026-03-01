Feature: Complete purchase on SauceDemo

  Scenario: User logs in, adds a product, and completes checkout
    Given I am on "https://www.saucedemo.com/"
    When I enter "standard_user" in the username field
    And I enter "secret_sauce" in the password field
    And I click the login button
    Then I should see the text "Products" on the page
    When I click "Add to cart" for "Sauce Labs Backpack"
    And I click the shopping cart icon
    Then I should see the text "Your Cart" on the page
    When I click the Checkout button
    Then I should see the text "Checkout: Your Information" on the page 
    When I enter "John" in the first name field
    And I enter "Doe" in the last name field
    And I enter "12345" in the zip code field
    And I click the Continue button
    Then I should see the text "Checkout: Overview" on the page 
    When I click the Finish button
    Then I should see the text "Thank you for your order!" on the page 
    When I click the Back Home button
    Then I should be on the products page
    Then I should see the text "Products" on the page
