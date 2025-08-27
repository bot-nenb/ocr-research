Read the contents of synthesis/UPDATED_DEMO_TASKS.md. Now create a detailed to-do list for building demo 1. This list should follow the following requirements. Install the Python environment using uv only. DO NOT USE pip. The demo will be built on a machine with Ubuntu 24.04 LTS operating system. It will only have access to the CPU when building the demo, and will not have access to the GPU. But you should design the demo so that it can also be run on GPU in the future. And for now just ignore the OLMoCR use-case as it requires GPU. Store the information in tasks/task-1/to-do.md.

Ok, now I want you to build steps 1-3 from the to-do list that you created. Let me know if you get stuck and I can help.

Ok, now I want you to build steps 4-5 from the to-do list that you created. Let me know if you get stuck and I can help.

Ok, now I want you to build step 6 from the to-do list that you created. Let me know if you get stuck and I can help. Remember that you should only use uv and not use pip. uv sync --extra dev --extra cpu --extra paddle will make sure that the environment is correct. And then uv run python ... will do the rest.

Now build steps 7-9. When you are finished let me know.

Now build steps 10-14. When you are finished let me know.

Review the code in the demo/ subdirectory. Focus on issues like incorrect logic inside of functions, tests that do not test anything, and places where the code deviates from steps 1-14 in the to-do document. DO NOT CHANGE any code during this review. Simple create a markdown document that outlines all the issues that you have found.

Read CODE_REVIEW.md. Write a Markdown document that explains how you will fix the Immediate Priority and High Priority problems identified in the code review. DO NOT MAKE code changes.

Now implement the fixes in FIX_IMPLEMENTATION_PLAN.md EXCEPT FOR i) Replace Threading with Multiprocessing and ii) Implement Device-Specific Worker Classes.

 > I would like you to add a visual comparison feature. Extend the HTML reports to include:
 - Original document images 
 - OCR text overlaid on images with bounding boxes
 - Side-by-side text comparison with highlighted differences  