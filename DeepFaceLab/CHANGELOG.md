# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),

and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

# DeepAscension

## [1.0.1]

## Added
- print web preview url to stdout

## Fixed
- add unicode support in the bat files so the logo works

## [1.0.0]

### Changed
- Lots of the Shennon stuff broke original features.  
Reversed almost all changes back to how it was in the MVE fork.  
I kept the following things:
  - The ME model
  - The Quick512 model
  - Super warp
  - dynamic xseg resolution
  - argi rg
  - ErrFaceFilter
  - alignment tools by marsmana1
  - alignment tools by yangala

## Added

- Improved xseg editor
- Superwarp in the AMP model

## Removed 

- Recent use section for bat scripts

# Shennong

## [3.0.1] - 23/05/2024
- Heavy update: Added an alternative training window provided by MVE with remote access and Linux with real-time preview and manipulation shortcuts.

- Checked bat files and other details for naming errors. Users are encouraged to provide feedback if any omissions are found. These interactions do not affect model quality.

- Resolved the issue of missing character display in the preview image, particularly the "????.jpg" in the bottom left corner. If any variables containing ".jpg" were identified, users were encouraged to report them for resolution.

## [3.0.0] - 22/05/2024
- Fixed models reporting errors for missing parameters after upgrading Shennong from other versions. This issue is handled by forcing to ask.

- Solved the problem where the summary would not display if parameters were missing, leading to a dead loop. Now, if a missing parameter is detected before displaying the summary, the user is prompted to set the parameter.

- Resolved the issue of Chinese characters becoming garbled in the preview window.

- Optimized the loss display in the preview window, ensuring that the dst is no longer directly overlaying the src, resulting in a blind spot. The intersection is now displayed in green, providing better visibility.

- Successfully tested new generation features: Xseg is now judged by the amount of hand-painted material and loss, while the face-swap model is judged by material amount and loss. Iteration number is no longer the primary reference.

- Training memory records will be restored in subsequent versions, serving as a diagnostic tool to determine whether the model can run. After reporting an error, users can simply take a screenshot of the summary to diagnose the issue.

- Future versions will introduce improvements to the pak selection process, allowing users to select from a list. Additionally, the loss of different paks will be recorded to provide feedback on the training stage. Password setting may also be implemented to protect material during remote training.

- Unified the Python environment into a single directory, laying the foundation for compiling some files into pyd. This step is particularly important for asset security, ensuring synchronization of software features across N card and A card environments.

## [2.6.1] - 17/05/2024
- Reintroduced the export dfm.bat for the Q224 model, which was inadvertently removed in later versions. Additionally, this version includes export dfm.bat for Q512.

- Internal testing for Q512 is now open. Users are invited to try it out, but please note that it's primarily suitable for one-on-one live special dan or video face swap, rather than universal dan.

- Process optimization: After selecting the model, the model summary now pops up first, followed by a 10-second interval to decide whether to modify.

- Expanded the preview image for Q512. Depending on user preference, the preview image format may also be changed for ME models in the future (5 columns instead of 2).

- Internal testing for WEBUI is underway. Users are advised not to click on the upper-right corner of the drop-down menu, as it may cause the model to collapse. Accessing port ip:6006 on the external network allows viewing previews, with a fixed refresh interval of 5 minutes (can be changed to 1 minute).

- Added numerous bat files, ensuring no omissions.

- The ME model is gradually transitioning to being named SN model. The next version is expected to include an automatic name change feature.

## [2.5.1] - 09/05/2024
- This update focuses on updating the Python environment, resulting in only one Python folder now, compared to the previous two. Additionally, the logic failure of config.txt in version 2.3.3 has been fixed, making the switch between N card and A card smoother.

- Fixed the old-SAEHD's txt to be saved in UTF-8 format, thanks to feedback from Taiwan compatriots regarding this issue.

- Addressed the parameter reversal in version 2.4.1 of the OLD trainer. Previously, GAN was put into the pre-training, resulting in no GAN option in the main training. Now, the pre-training does not show GAN. This optimization improves the UI experience without affecting model training and compatibility.

- Fixed missing file when synthesizing, causing an exception: Unable to load FaceEnhancer.npy. The file has been pasted back into place.

## [2.4.1] - 26/04/2024
- The Shinnon masking model can't be used directly but needs to be trained and saved before it can be utilized. It adds a resolution parameter and possibly others. If users want to use the Shinnon trained masks with the official original, ICE, or ME, they should train it with Shinnon and save it. Relying on scripting to write to the original mask will have the same effect as if it iterated once and then saved. If more parameters are added in the future, it will require more work to handle the original version.

- When synthesizing, the super score is missing files. Exception: Unable to load FaceEnhancer.npy. To address this issue, copy from _internal\DeepFaceLab\facelib to _internal\DeepFaceLab_old\facelib.

## [2.4.0] - 25/04/2024
- Previously, there was some limitation on masks other than 256. Upon reviewing the code today, it was realized that it was just a delayed release. Application and creation of Xseg models for other resolutions are now unlocked.
- The number of XS marker datasets owned has increased from 4000 to hundreds of thousands, with the potential to reach millions soon. Collaboration with Cxsmo has led to major breakthroughs so far.
- It's been a while since we released a full package. This time we added some tools in the eighth category bat and optimized the speed.
- Shennong netbook link comes with 224 SRC and DST material (again, took some time to organize and filter). I hope you try Quick224. Training takes no more than 2 days. Export the dfm and look at the effect, see if there is no flickering. I've also been testing the miniatures myself in recent days (so I've been a bit busy), using the attached footage as well.

## [2.3.3] - 23/04/2024
### Added
- Added some new tools. The original author can find me for points if he needs spirit stones. [Shennong MVE-DFL Chinese] v3.0.1

### Improved
- Changed various bat files that previously required selecting the graphics card and RG to automatically read the configuration from the txt.

### Removed
- Cleaned up some junk folders.

## [2.3.2] - 22/04/2024
### Added
- Added option: "Train SAEHD Continue from last time" in train Recent SAEHD.bat. Ignores the model and graphics card lists and parameters and starts training directly from the last session.

### Improved
- Record the GPU device selection and save it to options for future use (start training directly after selecting the model). This prevents the need to select the GPU every time. (Warm tip: when the model is small enough, dual cards are not as fast as a single card).

- Saved the graphics card and RG configurations of SAEHD to a local txt configuration file, which will be used by the "显卡设置 - RG开关" option. This harmonizes management and stops asking every time.

### Combined Features
- Combined features 1 and 3 to introduce a test effect: Double-click to automatically start training without needing to choose the model, graphics card, graphics card mode, or RG settings.


## [2.3.0] - 21/04/2024 (Completed, Not Passed On)
### Changed
- Planning for the 3.0 era is underway, starting with dfl-SAEHD changes this time. Combing model training scripts from scratch, omitting invalid parameters such as GAN, style, etc., which won't pop up in the early stages. In the official version, it will first select the graphics card, then select the model, then display the summary, and only modify the parameters at the end. This is a more logical sequence compared to selecting the model, selecting the card, modifying the parameters, and then displaying the summary. Additionally, the order of options and parameters will be organized, and the resolution will be placed at the beginning when creating a new Dan.

### Improved
- Improved the handling of option parameters to prevent situations where setting up parameters only to find them empty in aligned. This often occurs due to the need to replace and rename materials during training. Now, the program detects empty clips first, exits when necessary, and only shows options if the clip is non-empty. This feature involves compositing and exporting and will be optimized with pak naming in the future.

- Unified parameter queries to address frequent KeyError: 'retraining_samples' errors by forcing the program to ask directly instead of lazily relying on Enter. This ensures a smoother process and avoids errors.

### Added
- Quick224 in this version already supports the export of dfm. Quick512 and Quick384 are on the way!

## [2.2.0] - 15/04/2024 (Patch Issued)
### Improved
- Major overhaul of the XSeg application (Apply) to fix the old xs model naming and resolution fixed at 256. Also took into account the new model's multi-mask selection list and custom resolution. If there is only one model inside the mask folder (even if it's an old version of the town alt), it will be applied directly without popping up the selection list.

### Changed
- For the summary table of the XS model, I don't want to be judged by the number of iterations. After all, it can be modified and inherited. Instead, I would record the number of hand-labeled materials used for XS training and the current loss value. An incremental version number could also be recorded (learning ICE).

## [2.1.2] - 13/04/2024
### Fixed
- Corrected Python environment for exporting SAEHD model.dat.

### Updated
- Updated to override the official built-in Xseg model (including using it by default when synthesizing). Custom models are no longer necessary.

## [2.1.1] - 12/04/2024
### Fixed
- Handled ME synthesizer error reporting.

## [2.1.0] - 08/04/2024
### Changed
- In the ME new function [super twist], DFL is not necessary to add because itself has twisted more significantly.
- Introduction: ME and DFL original random distortion intensity is different; in fact, DFL distortion is greater. So even if the iteration milliseconds faster, the loss may drop more slowly, and DFL can be used as an alternative to DFL. This is also why people often say that DFL is more like SRC than ME.
- After tuning, if the parameter is too hard, it may only be suitable for practicing from scratch. (The magnitude is 100% larger, the unknown risk outweighs the benefit, and the implementation is abandoned).
- In order to be compatible with the original DFL base mold, ME with super distortion on is 20%~25% larger than DFL distortion but much larger than ME itself.
- Still providing visual data! ME rotates at [-2,2], DFL rotates at [-10,10], while ME super distorts at [-12,12]. In terms of scaling, ME's upper limit has always been slightly larger than DFL's. Perhaps one of the reasons why ME's BS is often 2 less than DFL's.

## [2.0.3] - 04/04/2024 (Uploaded)
### Improved
- Checked the SAEHD synthesis stage; too many modifications needed for multi-models. Temporarily specified default xseg256 for synthesizer.

### Fixed
- Fixed the "missing retrain loss" parameter error when the original version is converted to Shennong direct training. It will not report an error when plotting the table now, but it is recommended to walk through the parameter settings.
- Solved the problem of missing parameters of Xseg, Merge, Quick224. In fact, it's a bug that the links of drawing summary are different, which leads to the bug.

### Note
- Shennong 384 mask has collected thousands of manually labeled materials, ready for training. It is expected that hundreds of thousands of training materials will be added around June with the assistance of foreign friends. I'm willing to share the unified results as long as the netizens turn in their assignments. There will be no excuses of "I drew tens of thousands but the netizen only drew hundreds so it's only fair to give a mutilated version." Feel free to add to that! My personal results are the results shared by all! I haven't sold a penny of anything since I started (I can't cash out my spirit stones!), and of course, I'm in great need of getting donations or orders. I don't think any modeling results with substantial barriers should be sold for RMB, as they are immediately upsold to flood the market and devalue. Only a substitute trainer can lock up this source of means of production.

## [2.0.2] - 30/03/2024
### Improved
- Checked the SAEHD compositing stage for multi-masking applications, if time permits.

### Fixed
- Debugged the Quick224 model. Extreme training.

### Note
- Shennong 384 Shendan is playing well so far.


## [2.0.1] - 28/03/2024
### Known Issues
- The previous version of XSeg training implemented multi-model coexistence with selection by list index. However, testing revealed that the stage of applying the mask was missing the user selection process.
- There is a problem with the application and synthesis of non-256 XSeg models. It will take about a day to fix. This issue involves sensitive code (creation, writing, synthesis of non-256 XSeg models), and I'm going to protect it for a while until I get a strong enough 384 model to test.

### Changed
- Focused on modifying the write (apply) action of XSeg, which was not able to select the list in the original version and did not belong to the model base class. Rewrote the XSeg list index.
- Added a precautionary note: the synthesis stage may not have a list yet (copy code quickly). If this situation occurs, temporarily use the alt-po workaround.


## [2.0.0] - 27/03/2024
### Added
- XSeg now supports custom resolution and decoding dimensions. (This part of the code is not open for now, expected to be open source within 100 days)
  - [XSeg Masking] Good news! Create or retrain masks of any resolution (limited time).

- New Quick224 training and synthesis. Currently in the test phase. This feature aims to provide convenience, speed, and ease for newcomers.

### Fixed
- Previously, the summary only supported ME, leading to errors due to incorrect options. Now it has been categorized, with each architecture displaying a different table.

## [1.9.4] - 26/03/2024
### Known Issues
- Failed to change xseg resolution despite trying for 24 hours. This will be resolved in version 2.0.

### Changed
- SAEHD discarded the new version of the synthesizer. Users should delete the bat or avoid using it.
- SAEHD now uses the old version of the synthesizer, while ME uses the new version. They are one-to-one.

## [1.9.3] - 22/03/2024
### Improved
- Checked and aligned the table and layout of the original SAE architecture: [Shennong MVE-DFL Chinese Version] v3.0.1.

### Tested
- Finalized the performance of FP16: [New Reminder] [Shennong Hanhua] RG | DML | FP16 Very Important Measurements - [DFL] General Discussion - deepfacelab中文网 - Powered by Discuz! (dfldata.cc).

## [1.9.2] - 22/03/2024
### Changed
- Removed floating point to integer conversion in certain areas; both ME and SAEHD now use the latest OpenCV 4.7.

### Fixed
- Fully optimized the defects of the directory bat: [latest use] now will not duplicate copy itself.

### Added
- Added title to CMD window.
- DML and RG can now be manually controlled for switching.

### Improved
- Modified the degree of random distortion of ME and SAE during training to a compromise value. For instance, if ME is set to 2 and SAE to 10, they both now take 5.
- Based on incomplete evidence from the code, a 512 model requires 1024 cuts for ME training, and a 512 model requires 768 cuts for the original DFL training. This has not been thoroughly confirmed but is inferred from the scaling side of the function.

## [1.9.0] - 21/03/2024 (Advance Warning)
### Changed
- SAEHD's Graphics Settings - Hardware Accelerated GPU Program prompt has been changed to "Initialize.bat" to set it automatically. Removed expired and misleading image links.

- Completely phased out the DX12 environment in favor of the DML plugin. Now, AMD's ME and SAEHD are unified versions with CUDA. Prior to version 1.8.6, DX12 users were stuck on 1.5.4 due to differing codebases, making it troublesome to update A card support.

- Updated the summary table layout to include N card + SAEHD, A card + ME, and A card + SAEHD, in addition to N card + ME.

- Reduced the environment size by half to save space after decompression.

- Made RG bundling optional due to its significant impact:
  [Shennong Hanhua] RG (video memory optimization) and DML (A card) very important evaluation - [DFL] comprehensive discussion - deepfacelab中文网 - Powered by Discuz! (dfldata.cc)

### Added
- A card now also supports RG, as the DML interface is uniformly written with CUDA.

### Note
- Shennong's advantage is the 14-in-1 environment, which includes:
  - Official 20-series, official 30-series, official DX12, official DML.
  - Official 20-series RG, official 30-series RG, official DML RG.
  - ME20, ME30, ME DX12, ME DML, ME20 RG, ME30 RG, ME DML RG.
- This comprehensive integration is the magic of the Shennong Integration Pack.

## [1.8.6] - 18/03/2024
### Fixed
- Completed the missing half of the plugin for the patch that wasn't tested over the weekend.

### Changed
- After several rounds of modifications, updated true and false to whether or not.

### Notes
- Although the 1.8+ version only updated CUDA-ME's, the A-card and the original versions remain around 1.5.4. I personally didn't encounter many bugs.
- Decided to halt ME updates for now and focus on making the original version easier to use.

## [1.8.3] - 17/03/2024
### Fixed
- Patched the plug-in for the form to prevent errors reported due to missing forms, based on today's feedback.

### Changed
- Modified numerous typos and aligned symbols for better readability. This does not affect usability; users of version 1.8.0 need not worry.
- Updated true and false values to Chinese based on a suggestion, which I agree improves clarity.

## [1.8.0] - 16/03/2024
### Added
- Comprehensive reconstruction of the display format of the summary, including tidying up the txt format.

## [1.7.0] - 13/03/2024
### Added
- Fully modified yaml one-click training parameters, possibly enabling hot modification during training.
- Intentions to create a Chinese version of yaml, abandoning bilingual coexistence in the general classification, where specific keys must be in English. 
- Acknowledgement of the bilingual display in other areas, urging users to skillfully modify yaml or rely on the developer or a friend to create a UI.

## [1.6.2] - 28/02/2024
### Changed
- Modified the path error in the training bat

## [1.6.1] - 27/02/2024
### Added
- Sinicized 2 synthesizers. One of the original synthesizers used the picture of Cat's Chineseization.
- SAEHD now supports two synthesizers.

## [1.6.0] - 26/02/2024
### Fixed
- Fixed some errors in bat content and added some text introduction.

### Improved
- Simplified and merged the functions of "Model Training -- Train Models". For example, AMP no longer distinguishes between SRC-SRC to avoid misleading newcomers, essentially just modifying the path in bat.
- Automatically reads and writes configuration files, enabling one-click training. Merged them into regular training and replaced the 2-second wait with an inquiry.

### Added
- Determined that the A-card can use ME and SAEHD training in full. This version supports an all-in-one setup for 3 graphics card versions x 2 model architectures, with RG. Modified the ME branch of the A-card training code accordingly.
- Warm tip: cc-aug color migration mode is highly recommended for its effectiveness. However, this function can't be optimized with RG, so users may need to switch to DX12, even if they are using an N card.

## [1.5.4] - 05/02/2024
### Fixed
- Fixed a bug causing the bat menu to stretch in model application.
- Corrected the environment called when using the SAEHD face synthesis bat.
- Export dfm bat now links to the corresponding environment and corrects the model architecture name.

## [1.5.2] - 01/02/2024
### Added
- ME and SAEHD model training now fully supports A-card.
  - Note: ME's A-card training still needs testing; ensuring no errors is the initial step. Eye and mouth training are currently invalid and require testing assistance.

## [1.5.1] - 31/01/2024
### Fixed
- Corrected a path error in exporting dfm for three models.
- Resolved a bat error in ME synthesis.

## [1.5.0] - 30/01/2024
### Changed
- Updated environment; Landmarks auto-error recognition should no longer report errors.

### Fixed
- Resolved a long-standing issue where renaming the model caused errors. Implemented a prompt for the error while allowing the script to continue executing instead of terminating.

### Planned
- The next version aims to simplify the optional parameters for pre-training.

- In the upcoming version, efforts will be made to explicitly indicate whether OOM (Out of Memory) is due to a lack of video or virtual memory, providing a suggested hint. This aims to improve user-friendliness, as jumping out to several pages of code can be confusing for newcomers.

## [1.4.2] - 29/01/2024
### Added
- Continued to unlock some parameter restrictions in pre-training, such as RW (random warping), while disabling some parameters.

### Fixed
- Discovered that FP16 was not completely enabled. The missing part has been addressed.

### Planned
- In the next version, consider implementing guidance based on whether it's the first run or not.

### Note
- Reminder: If you are an A-card user, you may only be able to use SAEHD training after switching environments, and you can't use ME. If you are an A-card user, please chat with me privately to exchange ideas.

## [1.4.1] - 28/01/2024
### Added
- Removed the "pre-training" process, as it's not commonly used.
- According to the official default pre-training conversion, no longer forced to clear the inter and iteration number, but ask the user.
- Opened FP16 option. Users can now choose to enable FP16. Remember to enter "?" to see the description of this parameter.
- Explained in detail the role of each parameter -udtc. Please enter "?" in the training console for details.

### Fixed
- Due to the reopening of FP16, the GAN doesn't report errors anymore. However, the ME version of GAN is said to be able to select only 1 GPU number.
- The path in the bat of Landmarks auto-error detection was wrong. It has been corrected.

## [1.4.0] - 25/01/2024
### Major Update
- N cards support RG optimization! Smaller video memory can run bigger models, or increase BS cap. FP16 is still disabled, sacrificing effects for speed. Dev team partners want to work on mixed precision.

### Added
- ME and DFL both frameworks support N-card RG while A-card doesn't, tried it! DFL before in order to everyone unified can use, is to use the DX12 version, so the N card efficiency has a discount. Now it has been stripped away, and the driver type is selected by numbers 1 and 2 during training!
- This integration package combines 12 DFL versions in one. But no significant increase in the size of the installation package. (Includes original, ME, overlay whether RG or not, then multiply by 20 series 30 series and A-card three installers). (ME_A card version may have error when RG, please contact me, after all, A card users are too few).
- Built-in Landmarks automatic error recognition, aligned merge tool, the location of the bat. Category 8: --[ Other Tests -- Extra Function ]-- Below.

### Fixed
- Due to the lack of fp16 option, it causes the error of opening gan, so I forgot to fix it. Please see the top of the comment section for the solution.

## [1.3.2] - 18/01/2024
### Added
- Corrected some bat errors, added ME model compositing and exporting.
- Already working on how to minimize the compression of the training set. The current src is saved in the recovery data is very accumulated, can utilize it in the future!

### Fixed
- Fixed an unknown error: model renaming reported an error. This was solved by rolling back the version.

### Changed
- Last 1.3 version of SAEHD was not fully native, this time it is!
- Pre-training set has been moved to workspace because it often needs to be exchanged with the aligned folder.
- Xseg model has a separate directory in workspace. I don't want to mix it with the face model.

### Updated
- Have gone along with RO optimization, eye model, AVATAR model, multi-GPU (more than 3 cards) training. Also took some time to collect, the source code have got it all.

## [1.3.0] - 11/01/2024
### Added
- Solved the original SAEHD grafting BUG.
- Added a model conversion bat, currently only from the original version to upgrade ME!
  
### Changed
- Due to the DFL to ME is easy to ascending version is not easy to descending version, for the model name to do the distinction, lest misuse. SAEHD-ME is abbreviated as SAEME model or ME model!

## [1.2.3] - 10/01/2024
### Failed Version (Records Only)
- Implanted the original DFL model, but something went wrong.
- Continuing tomorrow, main error reported is EYE_MOUTH.
- The next version may rename the ME version of SAEHD to SAEME to avoid random conversion.
- The next version will add the conversion bat between original and ME.

## [1.2.2] - 09/01/2024
### Added
- Support for A card, and comes with a bat for switching graphics cards
- Provide option when initializing: can remove Buddha statue and cat

### Changed
- Starting from version 1.2.1, it's not a patch, but a complete integration package. Note: 10-series, 20-series, 30-series N cards; DX12 are all integrated into the same package.

### Removed
- Completely remove the fp16 option

### Fixed
- Ada locks up when upgrading from the original version
- Disabled ada option after creating a model (as in, tris can't be changed)

### Updated
- Keep the native environment (based on DX12 version) and add jsonschema and attr.

## [1.1.1] - 05/01/2024
### Changed
- Changed the environment pointed to by the bat of "export loss" to Python 3.9
- Added attr dependency to the Python 3.6.8 native environment

## [1.1.0] - 04/01/2024
### Added
- Change the welcome interface plus two cats
- 5 bat files to modify the wrong pre-training path “pretrain_Celeb”.
- Loss export (it's better to let everyone know than only a few people know)
- Python368 original environment has been supplemented with jsonschema
- Added a disclaimer

### Updated
- Update tensorflow and cuda, cudnn again (old graphics card did not speed up)

## [1.0.0] - 29/12/2023
### Initialized
- Usage: Overwrite to original installation directory
- Introduction: The original authors are Cioscos, seranus, Payuyi, AnkurSaini07 of the MachineEditor organization.
    - Note: keaidelaohu did not participate in the development of Me, he just copied the source code, changed the author's name without attribution, and encrypted it. This is a violation of the GPL 3.0 open-source agreement.
    - Note: In GitHub, searching through the [Deepfacelab] keywords can yield only one in a thousand results. And the ones you guys search out are likely to be the product of non-martiality! Because you guys don't know how to search for forked projects that follow the rules, like MachineEditor. Search keywords [deepfacelab fork:true] only then you can find the good stuff!
- MVE features: new parameters (including but not limited to):
    - [n] Use fp16 ( y/n ? :help ): Don't turn this on straight away, the model may crash!
    - [n] Eyes priority ( y/n ? :help ): separate eye training
    - [n] Mouth priority ( y/n ? :help ): Separate eye training. :help ) :separate mouth training
    - [SSIM] Loss function ( SSIM/MS-SSIM/MS-SSIM L1 ? :help ): This seems to compute the loss by the face similarity algorithm, search for the term yourself, just keep the default.
    - [5e-05] Learning rate ( 0.0 ... 1.0 ? :help ): Learning rate is usually left untouched, but you can control the learning rate down manually, for example, I change it to 3e-05 when the loss reaches 0.25.
    - [n] Enable random downsample of samples ( y/n ? :help ): Same as random distortion, enhance generalization: randomly reduce resolution.
    - [n] Enable random noise added to samples ( y/n ? :help ): Same as random distortion, enhance generalization: random downsampling. :help ) :Same as random distortion, generalization enhancement: random noise map
    - [n] Enable random blur of samples ( y/n ? :help ): Enable random blur of samples ( y/n ?help ) :Enable random blur of samples ( y/n ?help ) :Same as random distortion, enhance generalization
    - [n] Enable random jpeg compression of samples ( y/n ?help ): Same as random distortion, enhance generalization: random blur map. :help ) :Same as random distort, enhance generalization: random compression of image quality.
    - [none] Enable random shadows and highlights of samples: more advanced random, enhance generalization: random simulation of light and shadow training
    - Added light and shadow color learning algorithms: “fs-aug”, “cc-aug”
- Workload:
    1. Chinese localization
    2. Upgrade python3.68 to 3.9.18
    3. Upgraded tensorflow-gpu2.6.0(or 2.4.0) to tensorflow-gpu2.10.0
    4. Dynamic bat interaction script in the main directory
- GitHub Link: [curios-city/DeepFaceLab](https://github.com/curios-city/DeepFaceLab)
    - Note: The link has been put at the top for free, the spirit stone is just a voluntary reward!

## Future Version Previews:
1. Completed
2. Completed
3. Completed
8. Completed
9. Completed
4. First do a good job of training the various stages of preset yaml one-click training, and then in the future to make a fully automated hosting
5. Increase 256 or more resolution mask training, application, synthesis
6. Extreme compression of PAK files
7. New model “Shennong frame structure V1.0”.

# MachineEditor

## [1.2.0] - 10/02/2022
### Added
-  New custom saving time; default is 25 minutes - DeepFake ENG ITA
### Updated
 - Added random blur to shadow augmentation
 - [Splittable random shadows](https://github.com/MachineEditor/DeepFaceLab/commit/9bf3eb5851c6d0fbd2cea201332a60b047bb9113)
 - Added a handful of checks to indicate that the dataset is in zip or pak form when the functions used require a bulk dataset
### Fixed
- [true face power only avaible when gan power > 0](https://github.com/MachineEditor/DeepFaceLab/commit/c833e2a2642e993409018e1f92c2565739056024)
 - [wrong def. cpu cap](https://github.com/MachineEditor/DeepFaceLab/commit/18afb868bf486e4dd4bf5eba5d41fb14a5925620)
- Other random fixes

## [1.1.0] - 29/12/2021
### Added
 -  'random_hsv_power' from Official fork - seranus
 - New 'force_full_preview' to force to do not separate dst, src and pred views in different frames 	 - randomfaker
 ### Updated
 - Refactored two pass splitting it into 3 mode: None, face, face + mask - randomfaker
 - Updated shadow augmentation splitting it in: None, src, dst, all - DeepFake ENG ITA
 - [Update requirements-colab.txt](https://github.com/MachineEditor/DeepFaceLab/commit/bfaf6255ba5c70d831151099c67b65d87a9f5466)
### Fixed
- [config-training-file supports now files](https://github.com/MachineEditor/DeepFaceLab/commit/424469845960b06652af81e77409c70a6aa73003)

## [1.0.0] - 10/12/2021
### Initialized
We created this fork from several other forks of DeepFaceLab.
Many features of this fork comes mainly from [JH's fork](https://github.com/faceshiftlabs/DeepFaceLab).
#### Features from JH's fork
- [Web UI for training preview](doc/features/webui/README.md)
- [Random color training option](doc/features/random-color/README.md)
- [Background Power training option](doc/features/background-power/README.md)
- [MS-SSIM loss training option](doc/features/ms-ssim)
- [GAN label smoothing and label noise options](doc/features/gan-options)
- MS-SSIM+L1 loss function, based on ["Loss Functions for Image Restoration with Neural Networks"](https://research.nvidia.com/publication/loss-functions-image-restoration-neural-networks)
- Autobackup options:
	- Session name
	- ISO Timestamps (instead of numbered)
	- Max number of backups to keep (use "0" for unlimited)
- New sample degradation options (only affects input, similar to random warp):
	- Random noise (gaussian/laplace/poisson)
	- Random blur (gaussian/motion)
	- Random jpeg compression
	- Random downsampling
- New "warped" preview(s): Shows the input samples with any/all distortions.
#### Features from other forks
- FaceSwap-Aug in the color transfer modes
- Custom face types
#### Features from MVE Development team
- External configuration files by [Cioscos](https://github.com/Cioscos) aka DeepFake ENG ITA
	- use --auto_gen_config CLI param to auto generate config. file or resume its configuration
	- use --config_training_file CLI param external configuration file override
- Tensorboard support by [JanFschr](https://github.com/JanFschr) aka randomfaker
- AMP training updates - DeepFake ENG ITA & randomfaker
- shadow augmentation (needs testing to see if it can generalise well) - randomfaker
- filename labels by [Ognjen](https://github.com/seranus) aka JesterX aka seranus
- zip faceset support - randomfaker
- exposed new configuration parameters (cpu, lr, preview samples)
- Added pre-sharpen into the merger. It helps the model to fit better to the target face. Idea taken from [DeepFaceLive](https://github.com/iperov/DeepFaceLive)
- Added two pass option into the merger. It processes the generated face twice. Idea taken from [DeepFaceLive](https://github.com/iperov/DeepFaceLive)

[1.2.0]: https://github.com/MachineEditor/DeepFaceLab/tree/a7e0cbb0295ae35e9098ab383bc6e0a8bdd0f944
[1.1.0]: https://github.com/MachineEditor/DeepFaceLab/tree/bfaf6255ba5c70d831151099c67b65d87a9f5466
[1.0.0]: https://github.com/MachineEditor/DeepFaceLab/tree/6c5a5934452e174779561885fccf3f1ed38be9ae
