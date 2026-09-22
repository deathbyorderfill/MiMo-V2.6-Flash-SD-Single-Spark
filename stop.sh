#!/usr/bin/env bash
docker rm -f "${NAME:-mimo-sd}" >/dev/null 2>&1 && echo stopped || echo "not running"
