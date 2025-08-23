variable "project_id" {}
variable "topic_name" { default = "gmail-push" }
variable "domain" {}
variable "sa_email" {}

resource "google_pubsub_topic" "gmail" {
  name   = var.topic_name
  project = var.project_id
}

resource "google_pubsub_subscription" "gmail" {
  name  = "${var.topic_name}-sub"
  topic = google_pubsub_topic.gmail.name
  push_config {
    push_endpoint = "https://${var.domain}/gmail/webhook"
    oidc_token {
      service_account_email = var.sa_email
    }
  }
  ack_deadline_seconds = 10
}