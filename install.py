import launch
import os

# TODO: add pip dependency if need extra module only on extension

if not launch.is_installed("supabase"):
    launch.run_pip("install supabase", "requirements for EliAI-Engine")

if not launch.is_installed("boto3"):
    launch.run_pip("install boto3", "requirements for EliAI-Engine")