Feature: File Upload on ExpandTesting

  Scenario: User uploads a text file and verifies successful upload
    Given I am on "https://practice.expandtesting.com/upload"
    When I upload the file "C:\Users\MSI\OneDrive\Desktop\test-validation-system\tests\test_upload.txt" to the file input field
    When I click the "Upload" button
    Then I should see the text "File Uploaded!"