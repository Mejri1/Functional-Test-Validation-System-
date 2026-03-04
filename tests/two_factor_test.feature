Feature: Two-Factor Authentication on AuthenticationTest

  Scenario: User logs in with TOTP two-factor authentication
    Given I am on "https://authenticationtest.com/totpChallenge/"
    Then I should see the text "TOTP MFA Authentication Challenge"
    When I enter "totp@authenticationtest.com" in the email field
    And I enter "pa$$w0rd" in the password field
    And I click the "Login" button
    Then I should see the text "MFA Code"
    When I enter the two factor authentication code in the MFA field
    And I click the "Login" button
    Then I should see the text "Login Successful"