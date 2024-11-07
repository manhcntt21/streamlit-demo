Before you can send a request to the Speech-to-Text API, you must have completed the following actions. See the before you begin page for details.

    Enable Speech-to-Text on a Google Cloud project.
    Make sure billing is enabled for Speech-to-Text.

    Install the Google Cloud CLI, then initialize it by running the following command:

gcloud init

If you're using a local shell, then create local authentication credentials for your user account:

gcloud auth application-default login

You don't need to do this if you're using Cloud Shell.
(Optional) Create a new Google Cloud Storage bucket to store your audio data.

To avoid incurring charges to your Google Cloud account for the resources used on this page, follow these steps.

    Use the Google Cloud console to delete your project if you do not need it.



To authenticate to Speech-to-Text, set up Application Default Credentials. 