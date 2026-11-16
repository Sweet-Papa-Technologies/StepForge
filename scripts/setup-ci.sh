#!/usr/bin/env bash
#
# setup-ci.sh — idempotent setup of the release signing and branch protection
# for Sweet-Papa-Technologies/StepForge on GitHub (and the Azure side of the
# Windows signing).
#
#   ./scripts/setup-ci.sh --apple --env /path/to/.env   # Developer ID cert + notarization key
#   ./scripts/setup-ci.sh --azure                       # Trusted Signing: variables, environment, OIDC trust
#   ./scripts/setup-ci.sh --branch                      # protect main: PRs only, CI must pass
#   ./scripts/setup-ci.sh --show                        # what is configured right now
#
# --apple reads CERT_LOCATION (path to the Developer ID Application .p12) and
# CERT_PASS from the .env file, or takes --cert PATH and prompts for the password.
# The App Store Connect API key comes from the login keychain (service `asc-mcp`)
# or an AuthKey_*.p8 file.
#
set -euo pipefail

REPO="${REPO:-Sweet-Papa-Technologies/StepForge}"
AZURE_APP_ID="${AZURE_CLIENT_ID:-6ffedacb-25e4-49fc-9ccc-3379819fe8b3}"
ASC_KEY_ID="${ASC_KEY_ID:-45S6SMU5GY}"
ASC_ISSUER_ID="${ASC_ISSUER_ID:-08a7930e-9034-41cd-ab35-41afa2b19813}"
ASC_KEY_SERVICE="${ASC_KEY_SERVICE:-asc-mcp}"
ENV_FILE="" CERT="" DO_APPLE=0 DO_AZURE=0 DO_BRANCH=0 DO_SHOW=0

die()  { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }
info() { printf '\033[36m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[33mwarn:\033[0m %s\n' "$*" >&2; }

[ $# -gt 0 ] || { sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'; exit 0; }
while [ $# -gt 0 ]; do
    case "$1" in
        --apple)  DO_APPLE=1; shift ;;
        --azure)  DO_AZURE=1; shift ;;
        --branch) DO_BRANCH=1; shift ;;
        --show)   DO_SHOW=1; shift ;;
        --env)    ENV_FILE="${2:?}"; shift 2 ;;
        --cert)   CERT="${2:?}"; shift 2 ;;
        --repo)   REPO="${2:?}"; shift 2 ;;
        *) die "unknown option: $1" ;;
    esac
done
command -v gh >/dev/null || die "the GitHub CLI is not installed"
gh auth status >/dev/null 2>&1 || die "not logged in to GitHub: gh auth login"

if [ "$DO_SHOW" = 1 ]; then
    info "secrets";    gh secret list -R "$REPO" | sed 's/^/    /'
    info "variables";  gh variable list -R "$REPO" | sed 's/^/    /'
    info "environments"; gh api "repos/$REPO/environments" --jq '.environments[].name' | sed 's/^/    /'
    info "main protection"
    gh api "repos/$REPO/branches/main/protection" --jq '{checks: .required_status_checks.contexts, admins: .enforce_admins.enabled, reviews: .required_pull_request_reviews.required_approving_review_count, force_push: .allow_force_pushes.enabled}' 2>/dev/null | sed 's/^/    /' || echo "    (not protected)"
    exit 0
fi

if [ "$DO_APPLE" = 1 ]; then
    info "Apple: Developer ID certificate"
    PASS=""
    if [ -n "$ENV_FILE" ]; then
        [ -f "$ENV_FILE" ] || die "no such file: $ENV_FILE"
        CERT="$(sed -n 's/^CERT_LOCATION=//p' "$ENV_FILE" | tr -d '"' | head -1)"
        PASS="$(sed -n 's/^CERT_PASS=//p' "$ENV_FILE" | tr -d '"' | head -1)"
    fi
    CERT="${CERT/#\~/$HOME}"
    [ -f "$CERT" ] || die "no .p12 (pass --env FILE with CERT_LOCATION, or --cert PATH)"
    if [ -z "$PASS" ]; then printf ' ? the .p12 password: '; read -rs PASS; echo; fi
    # Probe the file in a throwaway keychain: wrong password or the wrong kind of
    # certificate fails here, not 20 minutes into a release.
    PROBE="$(mktemp -d)/probe.keychain-db"; KPASS=$(openssl rand -hex 16)
    security create-keychain -p "$KPASS" "$PROBE"
    security import "$CERT" -k "$PROBE" -P "$PASS" -T /usr/bin/codesign >/dev/null 2>&1 \
        || { security delete-keychain "$PROBE"; die "could not import the .p12 (wrong password?)"; }
    NAME=$(security find-identity -v -p codesigning "$PROBE" | grep -o '"Developer ID Application: [^"]*"' | head -1 | tr -d '"')
    security delete-keychain "$PROBE"
    [ -n "$NAME" ] || die "the .p12 holds no Developer ID Application identity"
    TEAM=$(printf '%s' "$NAME" | sed -n 's/.*(\([A-Z0-9]\{10\}\))$/\1/p')
    echo "    identity: $NAME"
    base64 -i "$CERT" | gh secret set APPLE_CERT_P12 -R "$REPO"
    printf '%s' "$PASS" | gh secret set APPLE_CERT_PASSWORD -R "$REPO"
    printf '%s' "$TEAM" | gh secret set APPLE_TEAM_ID -R "$REPO"

    info "Apple: notarization (App Store Connect API key $ASC_KEY_ID)"
    KEY="$(security find-generic-password -s "$ASC_KEY_SERVICE" -a "$ASC_KEY_ID" -w 2>/dev/null || true)"
    case "$KEY" in *BEGIN*) ;; "") ;; *) KEY="$(printf '%s' "$KEY" | xxd -r -p 2>/dev/null || true)" ;; esac
    if ! printf '%s' "$KEY" | grep -q "BEGIN PRIVATE KEY"; then
        F="$(ls "$HOME"/.appstoreconnect/private_keys/AuthKey_"$ASC_KEY_ID".p8 "$HOME"/Downloads/AuthKey_"$ASC_KEY_ID".p8 2>/dev/null | head -1 || true)"
        [ -n "$F" ] && KEY="$(cat "$F")"
    fi
    printf '%s' "$KEY" | grep -q "BEGIN PRIVATE KEY" || die "App Store Connect key $ASC_KEY_ID not found"
    printf '%s' "$KEY" | base64 | gh secret set ASC_KEY_P8 -R "$REPO"
    printf '%s' "$ASC_KEY_ID" | gh secret set ASC_KEY_ID -R "$REPO"
    printf '%s' "$ASC_ISSUER_ID" | gh secret set ASC_ISSUER_ID -R "$REPO"
    echo "    6 secrets set"
fi

if [ "$DO_AZURE" = 1 ]; then
    command -v az >/dev/null || die "the Azure CLI is not installed"
    ACC=$(az account show -o json) || die "not logged in to Azure: az login"
    TENANT=$(printf '%s' "$ACC" | python3 -c 'import json,sys; print(json.load(sys.stdin)["tenantId"])')
    SUB=$(printf '%s' "$ACC" | python3 -c 'import json,sys; print(json.load(sys.stdin)["id"])')
    info "Azure: repository variables"
    gh variable set AZURE_CLIENT_ID -R "$REPO" -b "$AZURE_APP_ID"
    gh variable set AZURE_TENANT_ID -R "$REPO" -b "$TENANT"
    gh variable set AZURE_SUBSCRIPTION_ID -R "$REPO" -b "$SUB"

    info "GitHub: 'release' environment, usable only from v* tags"
    gh api -X PUT "repos/$REPO/environments/release" --input - >/dev/null <<'JSON'
{"deployment_branch_policy": {"protected_branches": false, "custom_branch_policies": true}}
JSON
    if ! gh api "repos/$REPO/environments/release/deployment-branch-policies" --jq '.branch_policies[].name' | grep -qx 'v*'; then
        gh api -X POST "repos/$REPO/environments/release/deployment-branch-policies" \
            -f name='v*' -f type=tag >/dev/null
    fi

    info "Azure: trust GitHub OIDC tokens from the release environment"
    OWNER="${REPO%%/*}"; NAME_ONLY="${REPO##*/}"
    ORG_ID=$(gh api "orgs/$OWNER" --jq .id 2>/dev/null || gh api "users/$OWNER" --jq .id)
    REPO_ID=$(gh api "repos/$REPO" --jq .id)
    EXISTING=$(az ad app federated-credential list --id "$AZURE_APP_ID" --query "[].subject" -o tsv)
    for SUBJECT in "repo:$REPO:environment:release" "repo:$OWNER@$ORG_ID/$NAME_ONLY@$REPO_ID:environment:release"; do
        if printf '%s\n' "$EXISTING" | grep -qxF "$SUBJECT"; then
            echo "    already trusted: $SUBJECT"; continue
        fi
        CNAME="github-$(printf '%s' "$NAME_ONLY" | tr 'A-Z' 'a-z')-release-env"
        case "$SUBJECT" in *@*) CNAME="$CNAME-ids" ;; esac
        az ad app federated-credential create --id "$AZURE_APP_ID" --parameters "{
            \"name\": \"$CNAME\",
            \"issuer\": \"https://token.actions.githubusercontent.com\",
            \"subject\": \"$SUBJECT\",
            \"audiences\": [\"api://AzureADTokenExchange\"]}" >/dev/null
        echo "    trusted: $SUBJECT"
    done
fi

if [ "$DO_BRANCH" = 1 ]; then
    info "repository settings"
    gh api -X PATCH "repos/$REPO" -F allow_squash_merge=true -F allow_merge_commit=false \
        -F allow_rebase_merge=true -F delete_branch_on_merge=true -F allow_auto_merge=true >/dev/null
    info "protect main: pull requests only, CI must pass, no force push or deletion"
    gh api -X PUT "repos/$REPO/branches/main/protection" --input - >/dev/null <<'JSON'
{
  "required_status_checks": {"strict": true, "contexts": ["Engine tests", "Desktop app"]},
  "enforce_admins": true,
  "required_pull_request_reviews": {"required_approving_review_count": 0, "dismiss_stale_reviews": true},
  "restrictions": null,
  "allow_force_pushes": false,
  "allow_deletions": false,
  "required_conversation_resolution": true
}
JSON
    echo "    done (applies to admins too)"
fi
info "done"
